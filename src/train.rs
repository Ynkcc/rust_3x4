// code_files/src/train.rs
use crate::game_env::{DarkChessEnv, ACTION_SPACE_SIZE, Observation};
use crate::mcts::{Evaluator, MCTS, MCTSConfig, MctsNode};
use crate::nn_model::BanqiNet;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use tch::{nn, Device, Tensor};

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
        
        let probs = logits.softmax(-1, tch::Kind::Float);
        let probs_vec: Vec<f32> = Vec::from(probs.view([-1]));
        let value_scalar: f32 = f32::from(value);
        
        (probs_vec, value_scalar)
    }
}

pub fn train_loop() -> Result<()> {
    let device = Device::cuda_if_available();
    println!("Training using device: {:?}", device);
    
    let vs = nn::VarStore::new(device);
    let evaluator = Arc::new(NNEvaluator::new(&vs.root(), device));
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    
    // Training Hyperparameters
    let num_iterations = 10;
    let num_episodes = 20;
    let mcts_sims = 50;
    
    for iteration in 0..num_iterations {
        println!("Iteration {}", iteration);
        
        let mut examples = Vec::new();
        
        // 1. Self-Play
        for eps in 0..num_episodes {
            let mut env = DarkChessEnv::new();
            let mut mcts = MCTS::new(&env, evaluator.clone(), MCTSConfig { num_simulations: mcts_sims, cpuct: 1.0 });
            
            let mut episode_step = 0;
            let mut episode_data = Vec::new(); // (Observation, PolicyProbs, Player)
            
            loop {
                // Run MCTS
                mcts.run(&env);
                let probs = mcts.get_root_probabilities();
                
                // Store data
                episode_data.push((env.get_state(), probs.clone(), env.get_current_player()));
                
                // Select action (exploration vs exploitation)
                // For training, sample from probs
                let action = sample_action(&probs);
                
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
                            
                            // Backfill value to examples
                            for (obs, p, player) in episode_data {
                                let val = if player.val() == 1 { reward_red } else { -reward_red };
                                examples.push((obs, p, val));
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

fn sample_action(probs: &[f32]) -> usize {
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;
    let dist = WeightedIndex::new(probs).unwrap();
    let mut rng = thread_rng();
    dist.sample(&mut rng)
}

fn train_step(
    opt: &mut nn::Optimizer,
    net: &BanqiNet,
    examples: &[(Observation, Vec<f32>, f32)],
    batch_size: usize,
    device: Device
) -> f64 {
    // Dummy implementation: pick one batch
    if examples.is_empty() { return 0.0; }
    
    // Prepare tensors
    // Real implementation should shuffle and iterate over all data
    let (obs, target_probs, target_val) = &examples[0]; // just taking first for syntax check
    
    let board_tensor = Tensor::from_slice(&obs.board.into_raw_vec()).view([1, 16, 3, 4]).to(device);
    let scalar_tensor = Tensor::from_slice(&obs.scalars.to_vec()).view([1, 112]).to(device);
    let target_p = Tensor::from_slice(target_probs).view([1, 46]).to(device);
    let target_v = Tensor::from_slice(&[*target_val]).view([1, 1]).to(device);
    
    let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
    
    // Losses
    let p_loss = logits.log_softmax(-1, tch::Kind::Float).mul(&target_p).sum(tch::Kind::Float).neg();
    let v_loss = (value - target_v).mse_loss(tch::Reduction::Mean);
    
    let total_loss = p_loss + v_loss;
    
    opt.backward_step(&total_loss);
    
    f64::from(total_loss)
}

fn main() {
    // Entry point for training binary
    if let Err(e) = train_loop() {
        eprintln!("Training failed: {}", e);
    }
}