// parallel_train.rs - 并行自对弈训练系统
//
// 架构设计:
// - 主线程: 运行模型推理服务 (InferenceServer)
// - 工作线程池: 每个线程运行独立的自对弈游戏
// - 通信: 通过 channel 发送推理请求和接收结果
// - 批量推理: 收集多个请求后批量处理，提高GPU利用率

use banqi_3x4::game_env::{DarkChessEnv, Observation};
use banqi_3x4::mcts::{Evaluator, MCTS, MCTSConfig};
use banqi_3x4::nn_model::BanqiNet;
use anyhow::Result;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use rusqlite::{Connection, params};

// ================ 推理请求和响应 ================

/// 推理请求
#[derive(Debug)]
pub struct InferenceRequest {
    pub observation: Observation,
    pub action_masks: Vec<i32>,
    pub response_tx: mpsc::Sender<InferenceResponse>, // 每个请求携带自己的响应通道
}

/// 推理响应
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub policy: Vec<f32>,
    pub value: f32,
}

// ================ 批量推理服务器 ================

pub struct InferenceServer {
    vs: nn::VarStore,     // 持有 VarStore（包含模型权重）
    net: BanqiNet,        // 网络结构
    device: Device,
    request_rx: mpsc::Receiver<InferenceRequest>,
    batch_size: usize,
    batch_timeout_ms: u64,
}

impl InferenceServer {
    pub fn new(
        model_path: &str,
        device: Device,
        request_rx: mpsc::Receiver<InferenceRequest>,
        batch_size: usize,
        batch_timeout_ms: u64,
    ) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);
        let net = BanqiNet::new(&vs.root());
        
        // 加载模型权重
        vs.load(model_path)?;
        
        Ok(Self {
            vs,
            net,
            device,
            request_rx,
            batch_size,
            batch_timeout_ms,
        })
    }

    /// 运行推理服务（阻塞）
    pub fn run(&self) {
        println!("[InferenceServer] 启动，batch_size={}, timeout={}ms", 
            self.batch_size, self.batch_timeout_ms);
        
        let mut batch = Vec::new();
        let mut total_requests = 0;
        let mut total_batches = 0;
        let batch_timeout = Duration::from_millis(self.batch_timeout_ms);
        
        loop {
            // 尝试快速收集一批请求
            
            // 首先尝试非阻塞接收，快速收集可用的请求
            loop {
                match self.request_rx.try_recv() {
                    Ok(req) => {
                        batch.push(req);
                        total_requests += 1;
                        
                        // 如果达到批量大小，立即处理
                        if batch.len() >= self.batch_size {
                            break;
                        }
                    },
                    Err(mpsc::TryRecvError::Empty) => {
                        // 没有更多请求了
                        break;
                    },
                    Err(mpsc::TryRecvError::Disconnected) => {
                        // 所有发送者已断开
                        if !batch.is_empty() {
                            println!("[InferenceServer] 最终批次: {} 个请求", batch.len());
                            self.process_batch(&batch);
                            total_batches += 1;
                        }
                        println!("[InferenceServer] 所有客户端已断开，退出 (总计: {} 请求, {} 批次)", 
                            total_requests, total_batches);
                        return;
                    }
                }
            }
            
            // 如果收集到了请求，立即处理（不等待超时）
            if !batch.is_empty() {
                if total_batches % 100 == 0 {
                    println!("[InferenceServer] 处理批次#{}: {} 个请求", total_batches + 1, batch.len());
                }
                self.process_batch(&batch);
                total_batches += 1;
                batch.clear();
                continue;
            }
            
            // 如果没有请求，阻塞等待新请求（带超时）
            match self.request_rx.recv_timeout(batch_timeout) {
                Ok(req) => {
                    batch.push(req);
                    total_requests += 1;
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // 超时但没有请求，继续等待
                    continue;
                },
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    println!("[InferenceServer] 所有客户端已断开，退出 (总计: {} 请求, {} 批次)", 
                        total_requests, total_batches);
                    return;
                }
            }
        }
    }

    /// 批量处理推理请求
    fn process_batch(&self, batch: &Vec<InferenceRequest>) {
        if batch.is_empty() { return; }
        
        let start_time = Instant::now();
        let batch_len = batch.len();
        
        // 准备批量输入张量
        let mut board_data = Vec::new();
        let mut scalar_data = Vec::new();
        let mut mask_data = Vec::new();
        
        for req in batch {
            // Board: [STATE_STACK_SIZE, 8, 3, 4] -> flatten
            let board_flat: Vec<f32> = req.observation.board.as_slice().unwrap().to_vec();
            board_data.extend_from_slice(&board_flat);
            
            // Scalars: [STATE_STACK_SIZE * 56]
            let scalars_flat: Vec<f32> = req.observation.scalars.as_slice().unwrap().to_vec();
            scalar_data.extend_from_slice(&scalars_flat);
            
            // Masks: [46]
            let masks_f32: Vec<f32> = req.action_masks.iter().map(|&m| m as f32).collect();
            mask_data.extend_from_slice(&masks_f32);
        }
        
        // 构建张量: [batch, C, H, W]
        let board_tensor = Tensor::from_slice(&board_data)
            .view([batch_len as i64, 16, 3, 4])
            .to(self.device);
        
        let scalar_tensor = Tensor::from_slice(&scalar_data)
            .view([batch_len as i64, 112])
            .to(self.device);
        
        let mask_tensor = Tensor::from_slice(&mask_data)
            .view([batch_len as i64, 46])
            .to(self.device);
        
        // 前向推理
        let (logits, values) = tch::no_grad(|| {
            self.net.forward(&board_tensor, &scalar_tensor)
        });
        
        // 应用掩码并计算概率
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let probs = masked_logits.softmax(-1, Kind::Float);
        
        // 提取结果并发送响应到各自的通道
        for (i, req) in batch.iter().enumerate() {
            let policy_slice = probs.get(i as i64);
            let mut policy = vec![0.0f32; 46];
            policy_slice.to_device(Device::Cpu).copy_data(&mut policy, 46);
            
            let value = values.get(i as i64).squeeze().double_value(&[]) as f32;
            
            let response = InferenceResponse {
                policy,
                value,
            };
            
            // 发送响应到请求者的专属通道（忽略发送失败）
            let _ = req.response_tx.send(response);
        }
        
        let elapsed = start_time.elapsed();
        if batch_len >= 4 {  // 只在批量较大时输出日志
            println!("[InferenceServer] 批次处理: {} 个请求耗时 {:.2}ms", 
                batch_len, elapsed.as_secs_f64() * 1000.0);
        }
    }
}

// ================ Channel Evaluator（用于MCTS） ================

pub struct ChannelEvaluator {
    request_tx: mpsc::Sender<InferenceRequest>,
}

impl ChannelEvaluator {
    pub fn new(request_tx: mpsc::Sender<InferenceRequest>) -> Self {
        Self { request_tx }
    }
}

impl Evaluator for ChannelEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        // 为此次请求创建一次性响应通道
        let (response_tx, response_rx) = mpsc::channel();
        
        // 发送推理请求
        let req = InferenceRequest {
            observation: env.get_state(),
            action_masks: env.action_masks(),
            response_tx,
        };
        
        self.request_tx.send(req).expect("推理服务已断开");
        
        // 等待响应（阻塞）
        let resp = response_rx.recv().expect("推理服务无响应");
        
        (resp.policy, resp.value)
    }
}

// ================ 并行自对弈工作器 ================

/// 自对弈工作器
pub struct SelfPlayWorker {
    worker_id: usize,
    evaluator: Arc<ChannelEvaluator>,
    mcts_sims: usize,
}

impl SelfPlayWorker {
    pub fn new(
        worker_id: usize,
        evaluator: Arc<ChannelEvaluator>,
        mcts_sims: usize,
    ) -> Self {
        Self {
            worker_id,
            evaluator,
            mcts_sims,
        }
    }

    /// 运行一局自对弈游戏
    pub fn play_episode(&self, episode_num: usize) -> Vec<(Observation, Vec<f32>, f32, Vec<i32>)> {
        println!("  [Worker-{}] 开始第 {} 局游戏", self.worker_id, episode_num + 1);
        let start_time = Instant::now();
        
        let mut env = DarkChessEnv::new();
        let config = MCTSConfig { num_simulations: self.mcts_sims, cpuct: 1.0 };
        let mut mcts = MCTS::new(&env, self.evaluator.clone(), config);
        
        let mut episode_data = Vec::new();
        let mut step = 0;
        
        loop {
            // 运行MCTS
            mcts.run();
            let probs = mcts.get_root_probabilities();
            let masks = env.action_masks();
            
            // 保存数据
            episode_data.push((
                env.get_state(),
                probs.clone(),
                env.get_current_player(),
                masks,
            ));
            
            // 选择动作（前30步用温度采样增加探索）
            let temperature = if step < 30 { 1.2 } else { 0.8 };
            let action = sample_action(&probs, &env, temperature);
            
            // 执行动作
            match env.step(action, None) {
                Ok((_, _, terminated, truncated, winner)) => {
                    mcts.step_next(&env, action);
                    
                    if terminated || truncated {
                        // 分配奖励
                        let reward_red = match winner {
                            Some(1) => 1.0,
                            Some(-1) => -1.0,
                            _ => 0.0,
                        };
                        
                        let elapsed = start_time.elapsed();
                        println!("  [Worker-{}] 第 {} 局结束: {} 步, 胜者={:?}, 耗时 {:.1}s", 
                            self.worker_id, episode_num + 1, step, winner, elapsed.as_secs_f64());
                        
                        // 回填价值
                        let mut samples = Vec::new();
                        for (obs, p, player, mask) in episode_data {
                            let val = if player.val() == 1 { reward_red } else { -reward_red };
                            samples.push((obs, p, val, mask));
                        }
                        
                        return samples;
                    }
                },
                Err(e) => {
                    eprintln!("[Worker-{}] 游戏错误: {}", self.worker_id, e);
                    return Vec::new();
                }
            }
            
            step += 1;
            if step > 200 {
                // 超过最大步数，游戏平局
                println!("  [Worker-{}] 第 {} 局超时: {} 步", self.worker_id, episode_num + 1, step);
                let mut samples = Vec::new();
                for (obs, p, _, mask) in episode_data {
                    samples.push((obs, p, 0.0, mask));
                }
                return samples;
            }
        }
    }
}

/// 动作采样（带温度参数）
fn sample_action(probs: &[f32], env: &DarkChessEnv, temperature: f32) -> usize {
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;
    
    let non_zero_sum: f32 = probs.iter().sum();
    
    if non_zero_sum == 0.0 {
        // 回退：从有效动作中均匀选择
        let masks = env.action_masks();
        let valid_actions: Vec<usize> = masks.iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1 { Some(i) } else { None })
            .collect();
        
        let mut rng = thread_rng();
        *valid_actions.choose(&mut rng).expect("无有效动作")
    } else {
        // 应用温度参数
        let adjusted_probs: Vec<f32> = if temperature != 1.0 {
            let sum: f32 = probs.iter()
                .map(|&p| p.powf(1.0 / temperature))
                .sum();
            probs.iter()
                .map(|&p| p.powf(1.0 / temperature) / sum)
                .collect()
        } else {
            probs.to_vec()
        };
        
        let dist = WeightedIndex::new(&adjusted_probs).unwrap();
        let mut rng = thread_rng();
        dist.sample(&mut rng)
    }
}

// ================ 数据库操作（复用原有代码） ================

fn init_database(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    
    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            episode_type TEXT NOT NULL,
            board_state BLOB NOT NULL,
            scalar_state BLOB NOT NULL,
            policy_probs BLOB NOT NULL,
            value_target REAL NOT NULL,
            action_mask BLOB NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_iteration ON training_samples(iteration)",
        [],
    )?;
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episode_type ON training_samples(episode_type)",
        [],
    )?;
    
    println!("数据库初始化完成: {}", db_path);
    Ok(conn)
}

fn save_samples_to_db(
    conn: &mut Connection,
    iteration: usize,
    episode_type: &str,
    samples: &[(Observation, Vec<f32>, f32, Vec<i32>)]
) -> Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO training_samples 
             (iteration, episode_type, board_state, scalar_state, policy_probs, value_target, action_mask) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
        )?;
        
        for (obs, probs, value, mask) in samples {
            let board_bytes: Vec<u8> = obs.board.as_slice().unwrap()
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            let scalar_bytes: Vec<u8> = obs.scalars.as_slice().unwrap()
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            let probs_bytes: Vec<u8> = probs.iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            let mask_bytes: Vec<u8> = mask.iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            stmt.execute(params![
                iteration as i64,
                episode_type,
                board_bytes,
                scalar_bytes,
                probs_bytes,
                value,
                mask_bytes,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

fn load_samples_from_db(conn: &Connection) -> Result<Vec<(Observation, Vec<f32>, f32, Vec<i32>)>> {
    let mut stmt = conn.prepare(
        "SELECT board_state, scalar_state, policy_probs, value_target, action_mask 
         FROM training_samples"
    )?;
    
    let samples = stmt.query_map([], |row| {
        let board_bytes: Vec<u8> = row.get(0)?;
        let scalar_bytes: Vec<u8> = row.get(1)?;
        let probs_bytes: Vec<u8> = row.get(2)?;
        let value: f32 = row.get(3)?;
        let mask_bytes: Vec<u8> = row.get(4)?;
        
        let board_data: Vec<f32> = board_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let scalar_data: Vec<f32> = scalar_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let probs: Vec<f32> = probs_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let mask: Vec<i32> = mask_bytes.chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        use ndarray::Array;
        let board = Array::from_shape_vec((2, 8, 3, 4), board_data)
            .expect("Failed to reshape board data");
        let scalars = Array::from_vec(scalar_data);
        
        let obs = Observation { board, scalars };
        
        Ok((obs, probs, value, mask))
    })?;
    
    let mut result = Vec::new();
    for sample in samples {
        result.push(sample?);
    }
    
    Ok(result)
}

// ================ 训练步骤（复用原有代码） ================

fn train_step(
    opt: &mut nn::Optimizer,
    net: &BanqiNet,
    examples: &[(Observation, Vec<f32>, f32, Vec<i32>)],
    batch_size: usize,
    device: Device,
    epoch: usize,
) -> (f64, f64, f64) {
    if examples.is_empty() { return (0.0, 0.0, 0.0); }
    
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    let mut shuffled_examples = examples.to_vec();
    shuffled_examples.shuffle(&mut thread_rng());
    
    let mut total_loss_sum = 0.0;
    let mut policy_loss_sum = 0.0;
    let mut value_loss_sum = 0.0;
    let mut num_samples = 0;
    
    let policy_weight = 1.0 + (epoch as f32 * 0.2).min(2.0);
    
    for batch_start in (0..shuffled_examples.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(shuffled_examples.len());
        let batch = &shuffled_examples[batch_start..batch_end];
        
        for (obs, target_probs, target_val, masks) in batch.iter() {
            let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap()).view([1, 16, 3, 4]).to(device);
            let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap()).view([1, 112]).to(device);
            let target_p = Tensor::from_slice(target_probs).view([1, 46]).to(device);
            let target_v = Tensor::from_slice(&[*target_val]).view([1, 1]).to(device);
            
            let mask_vec: Vec<f32> = masks.iter().map(|&m| m as f32).collect();
            let mask_tensor = Tensor::from_slice(&mask_vec).view([1, 46]).to(device);
            
            let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
            
            let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
            let log_probs = masked_logits.log_softmax(-1, Kind::Float);
            
            let p_loss = (&target_p * &log_probs).sum(Kind::Float).neg() * (policy_weight as f64);
            let v_loss = value.mse_loss(&target_v, tch::Reduction::Mean);
            
            let total_loss = &p_loss + &v_loss;
            
            opt.backward_step(&total_loss);
            
            total_loss_sum += total_loss.double_value(&[]);
            policy_loss_sum += p_loss.double_value(&[]) / policy_weight as f64;
            value_loss_sum += v_loss.double_value(&[]);
            num_samples += 1;
        }
    }
    
    if num_samples > 0 { 
        (total_loss_sum / num_samples as f64,
         policy_loss_sum / num_samples as f64,
         value_loss_sum / num_samples as f64)
    } else { 
        (0.0, 0.0, 0.0)
    }
}

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
    
    // 并行配置
    let num_workers = (num_cpus::get() * 2).max(8); // 工作线程数：CPU核心数的2倍，至少8个
    let mcts_sims = 200; // 降低MCTS模拟次数以增加并发度
    let num_episodes_per_iteration = 100; // 增加游戏局数以补偿MCTS质量下降
    let inference_batch_size = 32.min(num_workers); // 推理批量大小
    let inference_timeout_ms = 5; // 批量推理超时（毫秒）- 进一步降低以提高响应速度
    
    println!("\n=== 并行训练配置 ===");
    println!("工作线程数: {}", num_workers);
    println!("每轮游戏数: {}", num_episodes_per_iteration);
    println!("MCTS模拟次数: {}", mcts_sims);
    println!("推理批量大小: {}", inference_batch_size);
    println!("推理超时: {}ms", inference_timeout_ms);
    
    // 初始化数据库
    let db_path = "training_samples.db";
    let mut conn = init_database(db_path)?;
    
    // 创建模型和优化器
    let vs = nn::VarStore::new(device);
    // **重要**: 立即创建网络以初始化所有参数
    let _init_net = BanqiNet::new(&vs.root());
    
    let learning_rate = 1e-4;
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;
    
    // 训练超参数
    let num_iterations = 200;
    let batch_size = 256;
    let epochs_per_iteration = 10;
    
    // 第一阶段：加载已有数据训练
    println!("\n=== 第一阶段：加载已有数据 ===");
    let existing_samples = load_samples_from_db(&conn)?;
    if !existing_samples.is_empty() {
        println!("加载了 {} 个样本", existing_samples.len());
        
        // 创建一个临时网络用于初始训练
        let temp_net = BanqiNet::new(&vs.root());
        
        for epoch in 0..20 {
            let (loss, p_loss, v_loss) = train_step(&mut opt, &temp_net, &existing_samples, batch_size, device, epoch);
            if (epoch + 1) % 5 == 0 {
                println!("  Epoch {}/20, Loss={:.4} (Policy={:.4}, Value={:.4})", 
                    epoch + 1, loss, p_loss, v_loss);
            }
        }
        
        vs.save("banqi_model_pretrained.ot")?;
        println!("已保存预训练模型");
    }
    
    // 第二阶段：并行自对弈训练
    println!("\n=== 第二阶段：并行自对弈训练 ===");
    
    for iteration in 0..num_iterations {
        println!("\n========== Iteration {}/{} ==========", iteration, num_iterations);
        
        // 创建推理通道
        let (req_tx, req_rx) = mpsc::channel::<InferenceRequest>();
        
        // 启动推理服务器线程 - 在线程中创建新的网络来避免所有权问题
        let temp_model_path = format!("banqi_model_iter_{}_temp.ot", iteration);
        vs.save(&temp_model_path)?;
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
        
        // 启动工作线程
        let mut worker_handles = Vec::new();
        let mut result_rxs = Vec::new();
        
        for worker_id in 0..num_workers {
            let req_tx_clone = req_tx.clone();
            let (result_tx, result_rx) = mpsc::channel();
            result_rxs.push(result_rx);
            
            let handle = thread::spawn(move || {
                let evaluator = Arc::new(ChannelEvaluator::new(req_tx_clone));
                let worker = SelfPlayWorker::new(worker_id, evaluator, mcts_sims);
                
                let mut all_samples = Vec::new();
                let episodes_per_worker = (num_episodes_per_iteration + num_workers - 1) / num_workers;
                
                for ep in 0..episodes_per_worker {
                    let samples = worker.play_episode(ep);
                    all_samples.extend(samples);
                }
                
                println!("  [Worker-{}] 完成所有 {} 局游戏", worker_id, episodes_per_worker);
                result_tx.send(all_samples).expect("无法发送结果");
            });
            
            worker_handles.push(handle);
        }
        
        // 关闭主请求发送端，以便推理服务器知道何时退出
        drop(req_tx);
        
        // 收集所有工作线程的结果
        let mut all_samples = Vec::new();
        for result_rx in result_rxs {
            if let Ok(samples) = result_rx.recv() {
                all_samples.extend(samples);
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
        
        println!("  收集了 {} 个训练样本", all_samples.len());
        
        // 保存样本到数据库
        save_samples_to_db(&mut conn, iteration, "self_play", &all_samples)?;
        
        println!("  开始训练...");
        
        // 训练模型
        let temp_net = BanqiNet::new(&vs.root());
        let mut total_losses = Vec::new();
        let mut policy_losses = Vec::new();
        let mut value_losses = Vec::new();
        
        let train_start = Instant::now();
        for epoch in 0..epochs_per_iteration {
            let (loss, p_loss, v_loss) = train_step(&mut opt, &temp_net, &all_samples, batch_size, device, epoch);
            total_losses.push(loss);
            policy_losses.push(p_loss);
            value_losses.push(v_loss);
            
            if (epoch + 1) % 2 == 0 {
                println!("  Epoch {}/{}, Loss={:.4} (Policy={:.4}, Value={:.4})", 
                    epoch + 1, epochs_per_iteration, loss, p_loss, v_loss);
            }
        }
        
        let train_elapsed = train_start.elapsed();
        let avg_loss: f64 = total_losses.iter().sum::<f64>() / total_losses.len() as f64;
        let avg_p_loss: f64 = policy_losses.iter().sum::<f64>() / policy_losses.len() as f64;
        let avg_v_loss: f64 = value_losses.iter().sum::<f64>() / value_losses.len() as f64;
        println!("  训练完成，耗时 {:.1}s，平均Loss: {:.4} (Policy={:.4}, Value={:.4})", 
            train_elapsed.as_secs_f64(), avg_loss, avg_p_loss, avg_v_loss);
        
        // 保存模型
        vs.save(format!("banqi_model_{}.ot", iteration))?;
        if iteration == num_iterations - 1 {
            vs.save("banqi_model_latest.ot")?;
        }
    }
    
    println!("\n训练完成！");
    Ok(())
}

fn main() {
    if let Err(e) = parallel_train_loop() {
        eprintln!("训练失败: {}", e);
    }
}
