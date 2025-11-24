// code_files/src/verify_scenario_2.rs

use banqi_3x4::game_env::DarkChessEnv;
use banqi_3x4::mcts::{Evaluator, MCTS, MCTSConfig, MctsNode};
use banqi_3x4::nn_model::BanqiNet;
use std::sync::Arc;
use tch::{nn, Device, Tensor, Kind};

// --- 未经训练的神经网络评估器 (同 verify.rs) ---
struct NNEvaluator {
    net: BanqiNet,
    device: Device,
}

impl NNEvaluator {
    fn new(vs: &nn::Path, device: Device) -> Self {
        Self { net: BanqiNet::new(vs), device }
    }
}

impl Evaluator for NNEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 16, 3, 4])
            .to(self.device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 112])
            .to(self.device);
        let (logits, value) = self.net.forward(&board_tensor, &scalar_tensor);

        // Apply action mask
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(self.device).view([1, 46]);
        let masked_logits = logits + (mask_tensor - 1.0) * 1e9;
        let probs = masked_logits.softmax(-1, Kind::Float);

        let probs_flat = probs.view([-1]);
        let probs_vec: Vec<f32> = (0..probs_flat.size()[0])
            .map(|i| probs_flat.double_value(&[i]) as f32)
            .collect();
        let value_scalar = value.squeeze().double_value(&[]) as f32;
        (probs_vec, value_scalar)
    }
}

// --- 提取 PUCT 最佳路径 ---
fn best_puct_path(root: &MctsNode, cpuct: f32, depth_limit: usize) -> Vec<(usize, f32, f32, f32, u32)> {
    let mut path = Vec::new();
    let mut current = root;
    let mut depth = 0;
    while depth < depth_limit && !current.children.is_empty() {
        let sqrt_total = (current.visit_count as f32).sqrt();
        let mut best: Option<(usize, f32, f32, f32, u32)> = None;
        for (&action, child) in &current.children {
            let q = child.q_value();
            let prior = child.prior;
            let puct = q + cpuct * prior * sqrt_total / (1.0 + child.visit_count as f32);
            let vc = child.visit_count;
            if let Some((_, _, _, _bpuct, _)) = best {
                if puct > _bpuct { best = Some((action, q, prior, puct, vc)); }
            } else {
                best = Some((action, q, prior, puct, vc));
            }
        }
        if let Some(record) = best {
            let action = record.0;
            path.push(record);
            current = current.children.get(&action).unwrap();
        } else {
            break;
        }
        depth += 1;
    }
    path
}

fn main() {
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };

    // 1. 初始化未经训练的网络
    let vs = nn::VarStore::new(device);
    let evaluator = Arc::new(NNEvaluator::new(&vs.root(), device));

    // 2. 初始化环境并设置特定场景
    let mut env = DarkChessEnv::new();
    // 调用我们在 game_env.rs 中新添加的方法
    env.setup_hidden_threats(); 

    println!("===== 自定义场景: 隐藏的威胁 =====");
    println!();
    env.print_board();
    println!("Hidden Pieces Pool: {:?}", env.hidden_pieces);
    println!("Reveal Probabilities: {:?}", env.get_reveal_probabilities());

    // 3. 运行 MCTS (1000 次模拟)
    let cpuct_val = 1.0;
    let config = MCTSConfig { cpuct: cpuct_val, num_simulations: 1000 };
    let mcts = MCTS::new(&env, evaluator.clone(), config);
    
    
    println!("\n===== MCTS 搜索完成 (1000 次模拟) =====");
    println!("根节点访问次数: {}", mcts.root.visit_count);

    // 4. 输出最佳 PUCT 路径
    let path = best_puct_path(&mcts.root, cpuct_val, 20);
    println!("\nPUCT 最高路径 (action, coords, Q, prior, PUCT, visits):");
    for (step_idx, (action, q, prior, puct, visits)) in path.iter().enumerate() {
        let coord_str = if let Some(coords) = env.get_coords_for_action(*action) {
            if coords.len() == 1 { format!("reveal@{}", coords[0]) } else { format!("move {}->{}", coords[0], coords[1]) }
        } else { "?".to_string() };
        println!(" Step {:02}: action {:02} ({}) | Q={:.4} prior={:.4} PUCT={:.4} visits={}" , step_idx, action, coord_str, q, prior, puct, visits);
    }
    
    // 最终叶节点价值
    if let Some((_, _, _, _, _)) = path.last() {
        let mut current = &mcts.root;
        for (a, _, _, _, _) in &path {
            current = current.children.get(a).unwrap();
        }
        println!("\n最终叶节点平均价值 Q: {:.4}", current.q_value());
    }

    // 根节点策略概率
    let probs = mcts.get_root_probabilities();
    println!("\n根节点策略概率 (前 20 个非零项):");
    for (i, p) in probs.iter().enumerate().filter(|(_,p)| **p > 0.0).take(20) {
        let coord_str = if let Some(coords) = env.get_coords_for_action(i) {
            if coords.len() == 1 { format!("reveal@{}", coords[0]) } else { format!("{}->{}", coords[0], coords[1]) }
        } else { "?".to_string() };
        println!("  action {:02} ({:<10}) prob={:.4}", i, coord_str, p);
    }

    println!("\n===== 验证结束 =====");
}