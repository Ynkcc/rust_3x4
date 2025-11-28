// parallel_train.rs - 并行自对弈训练系统主控制器
//
// 架构设计:
// - 主线程: 运行模型推理服务 (InferenceServer)
// - 工作线程池: 每个线程运行独立的自对弈游戏
// - 通信: 通过 channel 发送推理请求和接收结果
// - 批量推理: 收集多个请求后批量处理，提高GPU利用率

use banqi_3x4::nn_model::BanqiNet;
use banqi_3x4::inference::{InferenceServer, ChannelEvaluator};
use banqi_3x4::self_play::{SelfPlayWorker, ScenarioType};
use banqi_3x4::scenario_validation::validate_model_on_scenarios_with_net;
use banqi_3x4::training::train_step;
use banqi_3x4::game_env::Observation;
use anyhow::Result;
use std::sync::{Arc, mpsc};
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device};
use std::thread;

// ================ 主训练循环 ================

pub fn parallel_train_loop() -> Result<()> {
    // 设备配置
    let cuda_available = tch::Cuda::is_available();
    println!("CUDA available: {}", cuda_available);
    
    let device = if cuda_available {
        println!("Using CUDA device 0");
        Device::Cuda(0)
    } else {
        println!("Using CPU");
        Device::Cpu
    };
    
    // 训练配置
    let num_workers = 2; // 每个场景一个工作线程
    let mcts_sims = 800; // MCTS模拟次数
    let num_iterations = 20; // 训练迭代次数
    let num_episodes_per_iteration = 4; // 每轮每个场景的游戏数
    let inference_batch_size = 4;
    let inference_timeout_ms = 5;
    let batch_size = 32;
    let epochs_per_iteration = 5;
    let max_buffer_size = 1000;
    let learning_rate = 1e-3;
    
    println!("\n=== 场景自对弈训练配置 ===");
    println!("工作线程数: {} (每个场景一个)", num_workers);
    println!("每轮每场景游戏数: {}", num_episodes_per_iteration);
    println!("MCTS模拟次数: {}", mcts_sims);
    println!("训练迭代次数: {}", num_iterations);
    println!("推理批量大小: {}", inference_batch_size);
    println!("经验回放缓冲区: {}", max_buffer_size);
    println!("场景: TwoAdvisors, HiddenThreats");
    
    // 创建模型和优化器
    let vs = nn::VarStore::new(device);
    let net = BanqiNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;
    
    // 经验回放缓冲区
    let mut replay_buffer: Vec<(Observation, Vec<f32>, f32, Vec<i32>)> = Vec::new();
    
    // 主训练循环
    for iteration in 0..num_iterations {
        println!("\n========== Iteration {}/{} ==========", iteration + 1, num_iterations);
        
        // 保存临时模型供推理服务器使用
        let temp_model_path = format!("banqi_model_iter_{}_temp.ot", iteration);
        vs.save(&temp_model_path)?;
        
        // 创建推理通道
        let (req_tx, req_rx) = mpsc::channel();
        
        // 启动推理服务器线程
        let temp_model_path_clone = temp_model_path.clone();
        let inference_handle = thread::spawn(move || {
            match InferenceServer::new(
                &temp_model_path_clone,
                device,
                req_rx,
                inference_batch_size,
                inference_timeout_ms,
            ) {
                Ok(server) => server.run(),
                Err(e) => {
                    eprintln!("[InferenceServer] 初始化失败: {}", e);
                }
            }
        });
        
        // 启动工作线程 - 每个场景一个
        let scenarios = [ScenarioType::TwoAdvisors, ScenarioType::HiddenThreats];
        let mut worker_handles = Vec::new();
        let mut result_rxs = Vec::new();
        
        for (worker_id, scenario) in scenarios.iter().enumerate() {
            let req_tx_clone = req_tx.clone();
            let (result_tx, result_rx) = mpsc::channel();
            result_rxs.push(result_rx);
            let scenario_copy = *scenario;
            
            let handle = thread::spawn(move || {
                let evaluator = Arc::new(ChannelEvaluator::new(req_tx_clone));
                let worker = SelfPlayWorker::with_scenario(worker_id, evaluator, mcts_sims, scenario_copy);
                
                let mut all_episodes = Vec::new();
                for ep in 0..num_episodes_per_iteration {
                    let episode = worker.play_episode(ep);
                    all_episodes.push(episode);
                }
                
                println!("  [Worker-{}] 完成 {} 局 {} 游戏", 
                    worker_id, num_episodes_per_iteration, scenario_copy.name());
                result_tx.send(all_episodes).expect("无法发送结果");
            });
            
            worker_handles.push(handle);
        }
        
        // 关闭主请求发送端
        drop(req_tx);
        
        // 收集所有工作线程的结果
        let mut all_episodes = Vec::new();
        for result_rx in result_rxs {
            if let Ok(episodes) = result_rx.recv() {
                all_episodes.extend(episodes);
            }
        }
        
        // 等待所有工作线程完成
        for handle in worker_handles {
            handle.join().expect("工作线程异常");
        }
        
        // 等待推理服务器退出
        inference_handle.join().expect("推理服务器异常");
        
        // 清理临时模型文件
        let _ = std::fs::remove_file(&temp_model_path);
        
        // 过滤掉平局的游戏
        let filtered_episodes: Vec<_> = all_episodes.iter()
            .filter(|ep| ep.winner.is_some() && ep.winner.unwrap() != 0)
            .cloned()
            .collect();
        
        println!("  收集了 {} 局有胜负的游戏（共 {} 局）", 
            filtered_episodes.len(), all_episodes.len());
        
        // 提取样本
        let mut new_samples = Vec::new();
        for episode in &filtered_episodes {
            new_samples.extend(episode.samples.clone());
        }
        
        if new_samples.is_empty() {
            println!("  ⚠️ 本轮没有收集到有效样本，跳过训练");
            continue;
        }
        
        println!("  收集了 {} 个训练样本", new_samples.len());
        
        // 更新经验回放缓冲区
        replay_buffer.extend(new_samples);
        if replay_buffer.len() > max_buffer_size {
            let remove_count = replay_buffer.len() - max_buffer_size;
            replay_buffer.drain(0..remove_count);
        }
        println!("  经验回放缓冲区: {} 个样本", replay_buffer.len());
        
        // 训练
        println!("  开始训练...");
        let train_start = Instant::now();
        for epoch in 0..epochs_per_iteration {
            let (loss, p_loss, v_loss) = train_step(&mut opt, &net, &replay_buffer, batch_size, device, epoch);
            println!("    Epoch {}/{}: Loss={:.4} (Policy={:.4}, Value={:.4})", 
                epoch + 1, epochs_per_iteration, loss, p_loss, v_loss);
        }
        let train_elapsed = train_start.elapsed();
        println!("  训练完成，耗时 {:.1}s", train_elapsed.as_secs_f64());
        
        // 验证模型
        println!("\n  ========== 模型验证 ==========");
        let (scenario1, scenario2) = validate_model_on_scenarios_with_net(&net, device, iteration);
        println!("    场景1 (TwoAdvisors): a38={:.1}%, value={:.3}", 
            scenario1.masked_probs[38] * 100.0, scenario1.value);
        println!("    场景2 (HiddenThreats): a3={:.1}%, value={:.3}", 
            scenario2.masked_probs[3] * 100.0, scenario2.value);
        
        // 保存模型
        if (iteration + 1) % 5 == 0 || iteration == num_iterations - 1 {
            let model_path = format!("banqi_model_scenario_{}.ot", iteration + 1);
            vs.save(&model_path)?;
            println!("  已保存模型: {}", model_path);
        }
    }
    
    // 保存最终模型
    vs.save("banqi_model_scenario_latest.ot")?;
    println!("\n训练完成！已保存模型: banqi_model_scenario_latest.ot");
    println!("\n请使用以下命令测试模型:");
    println!("  cargo run --bin banqi-verify-trained -- banqi_model_scenario_latest.ot");
    
    Ok(())
}

fn main() {
    if let Err(e) = parallel_train_loop() {
        eprintln!("训练失败: {}", e);
    }
}
