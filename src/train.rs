// code_files/src/train.rs
use banqi_3x4::game_env::{DarkChessEnv, Observation};
use banqi_3x4::mcts::{Evaluator, MCTS, MCTSConfig};
use banqi_3x4::nn_model::BanqiNet;
use anyhow::Result;
use std::sync::Arc;
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};

// Wrapper for NN to implement Evaluator trait
struct NNEvaluator {
    net: BanqiNet,
    device: Device,
}

impl NNEvaluator {
    fn new(vs: &nn::Path, device: Device) -> Self {
        Self {
            net: BanqiNet::new(vs),
            device,
        }
    }
}

impl Evaluator for NNEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        // Convert Observation to Tensors
        let obs = env.get_state(); // Assuming get_state creates Observation
        
        // Board: [1, C, H, W]
        let board_tensor = Tensor::from_slice(&obs.board.into_raw_vec())
            .view([1, 16, 3, 4])
            .to(self.device);
            
        // Scalars: [1, F]
        let scalar_tensor = Tensor::from_slice(&obs.scalars.to_vec())
            .view([1, 112])
            .to(self.device);
            
        let (logits, value) = self.net.forward(&board_tensor, &scalar_tensor);
        
        // --- 修复：应用动作掩码 ---
        // 获取有效动作掩码
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(self.device).view([1, 46]);
        
        // 将掩码为 0 的位置的 logits 设为极小值 (-1e9)
        // 公式: masked_logits = logits + (mask - 1.0) * 1e9
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        
        let probs = masked_logits.softmax(-1, Kind::Float);
        // --- 修复结束 ---
        
        // Extract to Vec via data_ptr/try_data_ptr or use .shallow_clone() + as_slice
        let probs_flat = probs.view([-1]);
        let probs_vec: Vec<f32> = (0..probs_flat.size()[0])
            .map(|i| probs_flat.double_value(&[i]) as f32)
            .collect();
        let value_scalar: f32 = value.squeeze().double_value(&[]) as f32;
        
        (probs_vec, value_scalar)
    }
}

pub fn train_loop() -> Result<()> {
    // Check CUDA availability
    let cuda_available = tch::Cuda::is_available();
    let cuda_device_count = tch::Cuda::device_count();
    println!("CUDA available: {}", cuda_available);
    println!("CUDA device count: {}", cuda_device_count);
    
    let device = if cuda_available {
        println!("Using CUDA device 0");
        Device::Cuda(0)
    } else {
        println!("CUDA not available, using CPU");
        Device::Cpu
    };
    println!("Training using device: {:?}", device);
    
    let vs = nn::VarStore::new(device);
    let evaluator = Arc::new(NNEvaluator::new(&vs.root(), device));
    
    // 降低学习率以提高训练稳定性
    let learning_rate = 1e-4; // 从1e-3降到1e-4
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;
    
    println!("优化器配置: Adam, 学习率 = {}", learning_rate);
    
    // Training Hyperparameters (快速验证模式)
    let num_iterations = 5;  // 增加到5轮
    let num_episodes = 3;    // 每轮3局游戏
    let mcts_sims = 100;     // MCTS模拟次数
    let batch_size = 32;
    let epochs_per_iteration = 5; // 每次迭代训练5个epoch
    
    for iteration in 0..num_iterations {
        println!("\n============================================================");
        println!("Iteration {}", iteration);
        println!("============================================================");
        
        let mut examples = Vec::new();
        
        // 1. Self-Play
        for eps in 0..num_episodes {
            println!("  [Episode {}] 开始自我对弈...", eps);
            let mut env = DarkChessEnv::new();
            let mut mcts = MCTS::new(&env, evaluator.clone(), MCTSConfig { num_simulations: mcts_sims, cpuct: 1.0 });
            
            let mut episode_step = 0;
            let mut episode_data = Vec::new(); // (Observation, PolicyProbs, Player, ActionMasks)
            
            loop {
                // Run MCTS
                mcts.run();
                let probs = mcts.get_root_probabilities();
                
                // 调试：打印每步的MCTS策略分布（仅前几步和最后一步）
                if episode_step < 3 || episode_step % 50 == 0 {
                    println!("    [Step {}] MCTS策略分布 (Top 5):", episode_step);
                    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().enumerate()
                        .map(|(i, &p)| (i, p))
                        .filter(|(_, p)| *p > 0.0)
                        .collect();
                    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    for (i, &(action, prob)) in indexed_probs.iter().take(5).enumerate() {
                        println!("      #{}: action {} prob={:.4}", i+1, action, prob);
                    }
                }
                
                // Store data with action masks
                let masks = env.action_masks();
                episode_data.push((env.get_state(), probs.clone(), env.get_current_player(), masks));
                
                // Select action (exploration vs exploitation)
                // For training, sample from probs
                let action = sample_action(&probs, &env);
                
                // 调试：打印选择的动作
                if episode_step < 3 || episode_step % 50 == 0 {
                    println!("    [Step {}] 选择动作: {}", episode_step, action);
                }
                
                // Step Env
                match env.step(action, None) {
                    Ok((_, _, terminated, truncated, winner)) => {
                        // Advance MCTS root for reuse (optional in training, but good for robust testing)
                        mcts.step_next(&env, action);
                        
                        if terminated || truncated {
                            // Assign rewards
                            let reward_red = match winner {
                                Some(1) => 1.0,
                                Some(-1) => -1.0,
                                _ => 0.0,
                            };
                            
                            println!("    [Episode {} 结束] 总步数: {}, 胜者: {:?}, 奖励(红方视角): {:.1}", 
                                eps, episode_step, winner, reward_red);
                            
                            // Backfill value to examples (保留掩码)
                            for (obs, p, player, mask) in episode_data {
                                let val = if player.val() == 1 { reward_red } else { -reward_red };
                                examples.push((obs, p, val, mask));
                            }
                            break;
                        }
                    },
                    Err(e) => panic!("Error: {}", e),
                }
                
                episode_step += 1;
                if episode_step > 200 { 
                    println!("    [Episode {} 截断] 达到最大步数 200", eps);
                    break; 
                }
            }
        }
        
        println!("  [Self-Play 完成] 收集了 {} 个训练样本", examples.len());
        
        // 2. Training - 多个epoch遍历所有数据
        println!("  [Training] 开始训练，共 {} 个epoch", epochs_per_iteration);
        
        let mut total_losses = Vec::new();
        for epoch in 0..epochs_per_iteration {
            let loss = train_step(&mut opt, &evaluator.net, &examples, batch_size, device, epoch);
            total_losses.push(loss);
            println!("    Epoch {}: Loss = {:.4}", epoch, loss);
        }
        
        let avg_loss: f64 = total_losses.iter().sum::<f64>() / total_losses.len() as f64;
        println!("  [Training] 平均Loss: {:.4}", avg_loss);
        println!("  模型已保存: banqi_model_{}.ot", iteration);
        
        // 3. 训练后快速验证 - 测试模型在第一个样本上的表现
        if !examples.is_empty() {
            println!("  [Post-Training Validation]");
            let (obs, target_probs, target_val, masks) = &examples[0];
            
            let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap()).view([1, 16, 3, 4]).to(device);
            let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap()).view([1, 112]).to(device);
            
            let mask_vec: Vec<f32> = masks.iter().map(|&m| m as f32).collect();
            let mask_tensor = Tensor::from_slice(&mask_vec).view([1, 46]).to(device);
            
            let (logits, value) = evaluator.net.forward(&board_tensor, &scalar_tensor);
            let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
            let probs = masked_logits.softmax(-1, Kind::Float);
            
            let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
            let probs_vec: Vec<f32> = (0..46).map(|i| probs.double_value(&[0, i]) as f32).collect();
            
            // 计算KL散度来衡量策略匹配度
            let kl_div: f32 = target_probs.iter().zip(probs_vec.iter())
                .map(|(&t, &p)| if t > 0.0 { t * (t / (p + 1e-8)).ln() } else { 0.0 })
                .sum();
            
            println!("    第一个样本验证:");
            println!("      - Target Value: {:.4}, Predicted: {:.4}, Error: {:.4}", 
                target_val, value_pred, (target_val - value_pred).abs());
            println!("      - Policy KL散度: {:.4} (越小越好)", kl_div);
            
            // 显示Top-3对比
            let mut target_indexed: Vec<(usize, f32)> = target_probs.iter().enumerate()
                .filter(|(_, &p)| p > 0.0).map(|(i, &p)| (i, p)).collect();
            target_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let mut pred_indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate()
                .filter(|(_, p)| **p > 0.001).map(|(i, p)| (i, *p)).collect();
            pred_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            println!("      - Target Top-3: {}", 
                target_indexed.iter().take(3)
                    .map(|(a, p)| format!("{}({:.2}%)", a, p*100.0))
                    .collect::<Vec<_>>().join(", "));
            println!("      - Predicted Top-3: {}", 
                pred_indexed.iter().take(3)
                    .map(|(a, p)| format!("{}({:.2}%)", a, p*100.0))
                    .collect::<Vec<_>>().join(", "));
        }
        
        println!();
        
        // Save model every iteration for quick verification
        vs.save(format!("banqi_model_{}.ot", iteration))?;
        
        // Also save as "latest" for easy loading
        if iteration == num_iterations - 1 {
            vs.save("banqi_model_latest.ot")?;
        }
    }
    
    Ok(())
}

fn sample_action(probs: &[f32], env: &DarkChessEnv) -> usize {
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;
    
    // Filter out zero probabilities and check if we have any valid actions
    let non_zero_sum: f32 = probs.iter().sum();
    
    if non_zero_sum == 0.0 {
        // Fallback: choose uniformly from valid actions according to action mask
        let masks = env.action_masks();
        let valid_actions: Vec<usize> = masks.iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1 { Some(i) } else { None })
            .collect();
        
        let mut rng = thread_rng();
        *valid_actions.choose(&mut rng).expect("No valid actions available")
    } else {
        let dist = WeightedIndex::new(probs).unwrap();
        let mut rng = thread_rng();
        dist.sample(&mut rng)
    }
}

fn train_step(
    opt: &mut nn::Optimizer,
    net: &BanqiNet,
    examples: &[(Observation, Vec<f32>, f32, Vec<i32>)], // 增加掩码输入
    batch_size: usize,
    device: Device,
    epoch: usize, // 添加epoch参数用于调试
) -> f64 {
    if examples.is_empty() { return 0.0; }
    
    let verbose = epoch == 0; // 只在第一个epoch打印详细信息
    
    if verbose {
        println!("    [train_step] 训练样本数: {}, batch_size: {}", examples.len(), batch_size);
    }
    
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    let mut shuffled_examples = examples.to_vec();
    shuffled_examples.shuffle(&mut thread_rng());
    
    let mut total_loss_sum = 0.0;
    let mut num_batches = 0;
    
    // 遍历所有样本，每次处理batch_size个
    for batch_start in (0..shuffled_examples.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(shuffled_examples.len());
        let batch = &shuffled_examples[batch_start..batch_end];
        
        if verbose && batch_start == 0 {
            println!("    [train_step] 第一个batch (大小: {})", batch.len());
        }
        
        // 对batch中的每个样本进行训练
        for (sample_idx, (obs, target_probs, target_val, masks)) in batch.iter().enumerate() {
            // 只为第一个batch的第一个样本打印详细信息
            if verbose && batch_start == 0 && sample_idx == 0 {
                println!("      第一个样本:");
                println!("        - Target Value: {:.4}", target_val);
                println!("        - Target Policy (Top 3):");
                let mut indexed_probs: Vec<(usize, f32)> = target_probs.iter().enumerate()
                    .map(|(i, &p)| (i, p))
                    .filter(|(_, p)| *p > 0.0)
                    .collect();
                indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                for (i, &(action, prob)) in indexed_probs.iter().take(3).enumerate() {
                    println!("          #{}: action {} prob={:.4}", i+1, action, prob);
                }
            }
            
            // Prepare tensors
            let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap()).view([1, 16, 3, 4]).to(device);
            let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap()).view([1, 112]).to(device);
            let target_p = Tensor::from_slice(target_probs).view([1, 46]).to(device);
            let target_v = Tensor::from_slice(&[*target_val]).view([1, 1]).to(device);
            
            // 将掩码转换为 Tensor [1, 46]
            let mask_vec: Vec<f32> = masks.iter().map(|&m| m as f32).collect();
            let mask_tensor = Tensor::from_slice(&mask_vec).view([1, 46]).to(device);
            
            let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
            
            // 只为第一个样本打印预测信息
            if verbose && batch_start == 0 && sample_idx == 0 {
                let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
                println!("        - Predicted Value: {:.4}", value_pred);
                
                let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
                let probs_masked = masked_logits.softmax(-1, Kind::Float);
                let probs_masked_vec: Vec<f32> = (0..46)
                    .map(|i| probs_masked.double_value(&[0, i]) as f32)
                    .collect();
                let mut indexed_probs_masked: Vec<(usize, f32)> = probs_masked_vec.iter().enumerate()
                    .map(|(i, &p)| (i, p))
                    .filter(|(_, p)| *p > 0.001)
                    .collect();
                indexed_probs_masked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                println!("        - Predicted Policy (Top 3):");
                for (i, &(action, prob)) in indexed_probs_masked.iter().take(3).enumerate() {
                    println!("          #{}: action {} prob={:.4}", i+1, action, prob);
                }
            }
            
            // --- 应用动作掩码到训练 Loss 计算 ---
            let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
            let log_probs = masked_logits.log_softmax(-1, Kind::Float);
            
            let p_loss = (&target_p * &log_probs).sum(Kind::Float).neg();
            let v_loss = value.mse_loss(&target_v, tch::Reduction::Mean);
            
            let total_loss = &p_loss + &v_loss;
            
            // 只为第一个样本打印loss分解
            if verbose && batch_start == 0 && sample_idx == 0 {
                let p_loss_val = p_loss.double_value(&[]);
                let v_loss_val = v_loss.double_value(&[]);
                let total_loss_val = total_loss.double_value(&[]);
                
                println!("        Loss分解:");
                println!("          - Policy Loss: {:.4}", p_loss_val);
                println!("          - Value Loss: {:.4}", v_loss_val);
                println!("          - Total Loss: {:.4}", total_loss_val);
            }
            
            opt.backward_step(&total_loss);
            
            total_loss_sum += total_loss.double_value(&[]);
            num_batches += 1;
        }
    }
    
    let avg_loss = if num_batches > 0 { total_loss_sum / num_batches as f64 } else { 0.0 };
    
    if verbose {
        println!("    [train_step] 训练了 {} 个样本，平均Loss: {:.4}", num_batches, avg_loss);
    }
    
    avg_loss
}

fn main() {
    // Entry point for training binary
    if let Err(e) = train_loop() {
        eprintln!("Training failed: {}", e);
    }
}