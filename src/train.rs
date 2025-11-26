// code_files/src/train.rs
use banqi_3x4::game_env::{DarkChessEnv, Observation, Player};
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

// 验证函数：在两个标准场景上测试模型
fn validate_scenarios(net: &BanqiNet, device: Device) {
    use banqi_3x4::game_env::Player;
    
    // 场景1: 仅剩 R_A 与 B_A
    {
        let mut env = DarkChessEnv::new();
        env.setup_two_advisors(Player::Black);
        
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 16, 3, 4])
            .to(device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 112])
            .to(device);
        
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(device).view([1, 46]);
        
        let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let probs = masked_logits.softmax(-1, Kind::Float);
        
        let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
        let probs_vec: Vec<f32> = (0..46).map(|i| probs.double_value(&[0, i]) as f32).collect();
        
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate()
            .filter(|(_, &p)| p > 0.001)
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("  场景1 (R_A vs B_A):");
        println!("    Value: {:.4}", value_pred);
        println!("    Policy Top-3: {}", 
            indexed.iter().take(3)
                .map(|(a, p)| format!("action{}({:.1}%)", a, p*100.0))
                .collect::<Vec<_>>().join(", "));
        println!("    目标: action38(9->5) 应该概率最高 (MCTS结果: 99.34%)");
    }
    
    // 场景2: 隐藏的威胁
    {
        let mut env = DarkChessEnv::new();
        env.setup_hidden_threats();
        
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 16, 3, 4])
            .to(device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 112])
            .to(device);
        
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(device).view([1, 46]);
        
        let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let probs = masked_logits.softmax(-1, Kind::Float);
        
        let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
        let probs_vec: Vec<f32> = (0..46).map(|i| probs.double_value(&[0, i]) as f32).collect();
        
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate()
            .filter(|(_, &p)| p > 0.001)
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("  场景2 (Hidden Threat):");
        println!("    Value: {:.4}", value_pred);
        println!("    Policy Top-2: {}", 
            indexed.iter().take(2)
                .map(|(a, p)| format!("action{}({:.1}%)", a, p*100.0))
                .collect::<Vec<_>>().join(", "));
        println!("    目标: action3(reveal@3) 应该概率最高 (MCTS结果: 98.02%)");
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
    
    // Training Hyperparameters
    let num_iterations = 100;      // 正式训练：100轮迭代
    let num_episodes = 10;         // 每轮10局游戏
    let mcts_sims = 200;           // MCTS模拟次数
    let batch_size = 64;           // 增大batch size
    let epochs_per_iteration = 10; // 每次迭代训练10个epoch
    let validation_interval = 5;   // 每5轮进行一次验证
    
    for iteration in 0..num_iterations {
        println!("\n============================================================");
        println!("Iteration {}/{}", iteration, num_iterations);
        println!("============================================================");
        
        let mut examples = Vec::new();
        
        // 1. Self-Play
        for eps in 0..num_episodes {
            let mut env = DarkChessEnv::new();
            let mut mcts = MCTS::new(&env, evaluator.clone(), MCTSConfig { num_simulations: mcts_sims, cpuct: 1.0 });
            
            let mut episode_step = 0;
            let mut episode_data = Vec::new(); // (Observation, PolicyProbs, Player, ActionMasks)
            
            loop {
                // Run MCTS
                mcts.run();
                let probs = mcts.get_root_probabilities();
                
                // Store data with action masks
                let masks = env.action_masks();
                episode_data.push((env.get_state(), probs.clone(), env.get_current_player(), masks));
                
                // Select action (exploration vs exploitation)
                let action = sample_action(&probs, &env);
                
                // Step Env
                match env.step(action, None) {
                    Ok((_, _, terminated, truncated, winner)) => {
                        // Advance MCTS root for reuse
                        mcts.step_next(&env, action);
                        
                        if terminated || truncated {
                            // Assign rewards
                            let reward_red = match winner {
                                Some(1) => 1.0,
                                Some(-1) => -1.0,
                                _ => 0.0,
                            };
                            
                            // Backfill value to examples
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
                    break; 
                }
            }
            
            // 每完成一局游戏打印进度
            if (eps + 1) % 5 == 0 || eps == num_episodes - 1 {
                println!("  Self-play progress: {}/{} episodes", eps + 1, num_episodes);
            }
        }
        
        println!("  收集了 {} 个训练样本", examples.len());
        
        // 2. Training - 多个epoch遍历所有数据
        let mut total_losses = Vec::new();
        for epoch in 0..epochs_per_iteration {
            let loss = train_step(&mut opt, &evaluator.net, &examples, batch_size, device, epoch);
            total_losses.push(loss);
            
            // 每2个epoch或最后一个epoch打印进度
            if (epoch + 1) % 2 == 0 || epoch == epochs_per_iteration - 1 {
                println!("  Training progress: Epoch {}/{}, Loss = {:.4}", epoch + 1, epochs_per_iteration, loss);
            }
        }
        
        let avg_loss: f64 = total_losses.iter().sum::<f64>() / total_losses.len() as f64;
        println!("  平均Loss: {:.4}", avg_loss);
        
        // 3. 定期验证 - 使用两个标准场景测试模型
        if iteration % validation_interval == 0 || iteration == num_iterations - 1 {
            println!("\n  ========== 模型验证 (Iteration {}) ==========", iteration);
            validate_scenarios(&evaluator.net, device);
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
    examples: &[(Observation, Vec<f32>, f32, Vec<i32>)],
    batch_size: usize,
    device: Device,
    _epoch: usize, // 用于将来可能的epoch特定逻辑
) -> f64 {
    if examples.is_empty() { return 0.0; }
    
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    let mut shuffled_examples = examples.to_vec();
    shuffled_examples.shuffle(&mut thread_rng());
    
    let mut total_loss_sum = 0.0;
    let mut num_samples = 0;
    
    // 遍历所有样本，每次处理batch_size个
    for batch_start in (0..shuffled_examples.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(shuffled_examples.len());
        let batch = &shuffled_examples[batch_start..batch_end];
        
        // 对batch中的每个样本进行训练
        for (obs, target_probs, target_val, masks) in batch.iter() {
            // Prepare tensors
            let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap()).view([1, 16, 3, 4]).to(device);
            let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap()).view([1, 112]).to(device);
            let target_p = Tensor::from_slice(target_probs).view([1, 46]).to(device);
            let target_v = Tensor::from_slice(&[*target_val]).view([1, 1]).to(device);
            
            // 将掩码转换为 Tensor [1, 46]
            let mask_vec: Vec<f32> = masks.iter().map(|&m| m as f32).collect();
            let mask_tensor = Tensor::from_slice(&mask_vec).view([1, 46]).to(device);
            
            let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
            
            // 应用动作掩码到训练 Loss 计算
            let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
            let log_probs = masked_logits.log_softmax(-1, Kind::Float);
            
            let p_loss = (&target_p * &log_probs).sum(Kind::Float).neg();
            let v_loss = value.mse_loss(&target_v, tch::Reduction::Mean);
            
            let total_loss = &p_loss + &v_loss;
            
            opt.backward_step(&total_loss);
            
            total_loss_sum += total_loss.double_value(&[]);
            num_samples += 1;
        }
    }
    
    if num_samples > 0 { total_loss_sum / num_samples as f64 } else { 0.0 }
}

fn main() {
    // Entry point for training binary
    if let Err(e) = train_loop() {
        eprintln!("Training failed: {}", e);
    }
}