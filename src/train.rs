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
        let masked_logits = logits + (mask_tensor - 1.0) * 1e9;
        
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
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    
    // Training Hyperparameters
    let num_iterations = 2;
    let num_episodes = 5;
    let mcts_sims = 20;
    
    for iteration in 0..num_iterations {
        println!("Iteration {}", iteration);
        
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
                // For training, sample from probs
                let action = sample_action(&probs, &env);
                
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
                if episode_step > 200 { break; }
            }
            if eps % 5 == 0 { println!("  Episode {} finished.", eps); }
        }
        
        // 2. Training
        let batch_size = 32;
        // Simple shuffling and batching would go here
        // For brevity, we perform a few update steps
        
        let loss = train_step(&mut opt, &evaluator.net, &examples, batch_size, device);
        println!("  Loss: {:.4}", loss);
        
        // Save model
        if iteration % 5 == 0 {
            vs.save(format!("banqi_model_{}.ot", iteration))?;
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
    _batch_size: usize,
    device: Device
) -> f64 {
    // Dummy implementation: pick one batch
    if examples.is_empty() { return 0.0; }
    
    // Prepare tensors
    // Real implementation should shuffle and iterate over all data
    let (obs, target_probs, target_val, masks) = &examples[0]; // 解构时包含掩码
    
    let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap()).view([1, 16, 3, 4]).to(device);
    let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap()).view([1, 112]).to(device);
    let target_p = Tensor::from_slice(target_probs).view([1, 46]).to(device);
    let target_v = Tensor::from_slice(&[*target_val]).view([1, 1]).to(device);
    
    // 将掩码转换为 Tensor [1, 46]
    let mask_vec: Vec<f32> = masks.iter().map(|&m| m as f32).collect();
    let mask_tensor = Tensor::from_slice(&mask_vec).view([1, 46]).to(device);
    
    let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
    
    // --- 修复：应用动作掩码到训练 Loss 计算 ---
    // 屏蔽无效动作的 Logits
    let masked_logits = logits + (mask_tensor - 1.0) * 1e9;
    let log_probs = masked_logits.log_softmax(-1, Kind::Float);
    // --- 修复结束 ---
    
    let p_loss = (target_p * log_probs).sum(Kind::Float).neg();
    let v_loss = value.mse_loss(&target_v, tch::Reduction::Mean);
    
    let total_loss = p_loss + v_loss;
    
    opt.backward_step(&total_loss);
    
    total_loss.double_value(&[])
}

fn main() {
    // Entry point for training binary
    if let Err(e) = train_loop() {
        eprintln!("Training failed: {}", e);
    }
}