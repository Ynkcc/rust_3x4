// code_files/src/mcts.rs

use crate::{DarkChessEnv, Player, ACTION_SPACE_SIZE, Piece, PieceType, REVEAL_ACTIONS_COUNT, Slot};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// 1. 节点定义 (Node Definition)
// ============================================================================

/// MCTS 树节点
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// 访问次数 (N)
    pub visit_count: u32,
    /// 价值总和 (W)
    pub value_sum: f32,
    /// 先验概率 (P)
    pub prior: f32,
    /// 当前节点的动作-子节点映射 (针对 State Node)
    /// Key: Action Index
    pub children: HashMap<usize, MctsNode>,
    /// 标记是否已扩展
    pub is_expanded: bool,
    /// 该节点对应的玩家
    pub player: Player,

    // --- Chance Node 相关属性 ---
    /// 是否为机会节点 (Chance Node)
    pub is_chance_node: bool,
    /// 可能的状态映射 (针对 Chance Node)
    /// Key: Outcome ID (表示具体的翻棋结果), Value: (Probability, ChildNode)
    pub possible_states: HashMap<usize, (f32, MctsNode)>,
}

impl MctsNode {
    pub fn new(prior: f32, player: Player, is_chance_node: bool) -> Self {
        Self {
            visit_count: 0,
            value_sum: 0.0,
            prior,
            children: HashMap::new(),
            is_expanded: false,
            player,
            is_chance_node,
            possible_states: HashMap::new(),
        }
    }

    /// 获取平均价值 Q(s, a)
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}

// 辅助函数：为翻开的棋子生成唯一 ID
// 0-2: Red Sol, Adv, Gen; 3-5: Black Sol, Adv, Gen
fn get_outcome_id(piece: &Piece) -> usize {
    let type_idx = match piece.piece_type {
        PieceType::Soldier => 0,
        PieceType::Advisor => 1,
        PieceType::General => 2,
    };
    let player_offset = match piece.player {
        Player::Red => 0,
        Player::Black => 3,
    };
    type_idx + player_offset
}

// ============================================================================
// 2. 评估接口 (Evaluation Interface)
// ============================================================================

pub trait Evaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32);
}

pub struct RandomEvaluator;

impl Evaluator for RandomEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut probs = vec![0.0; ACTION_SPACE_SIZE];
        let masks = env.action_masks();
        let valid_count = masks.iter().sum::<i32>() as f32;
        
        if valid_count > 0.0 {
            for (i, &m) in masks.iter().enumerate() {
                if m == 1 {
                    probs[i] = 1.0 / valid_count;
                }
            }
        }
        let value: f32 = rng.gen_range(-1.0..1.0);
        (probs, value)
    }
}

// ============================================================================
// MCTS 主逻辑
// ============================================================================

pub struct MCTSConfig {
    pub cpuct: f32,
    pub num_simulations: usize,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            cpuct: 1.0,
            num_simulations: 50,
        }
    }
}

pub struct MCTS<E: Evaluator> {
    pub root: MctsNode, // made public for debug access if needed
    evaluator: Arc<E>,
    config: MCTSConfig,
}

impl<E: Evaluator> MCTS<E> {
    pub fn new(env: &DarkChessEnv, evaluator: Arc<E>, config: MCTSConfig) -> Self {
        let root = MctsNode::new(1.0, env.get_current_player(), false);
        Self {
            root,
            evaluator,
            config,
        }
    }

    /// 支持搜索树复用：根据动作将根节点推进一步
    pub fn step_next(&mut self, env: &DarkChessEnv, action: usize) {
        if let Some(mut child) = self.root.children.remove(&action) {
            if child.is_chance_node {
                // 如果是 Chance Node，说明上一步动作是翻棋
                // 我们需要检查当前环境实际翻出了什么棋子，从而选择正确的子节点
                
                // 获取动作对应的位置（假设 action < 12 时 action 即为位置）
                let sq = action; 
                let slot = &env.get_board_slots()[sq];
                
                match slot {
                    Slot::Revealed(piece) => {
                        let outcome_id = get_outcome_id(piece);
                        if let Some((_, next_node)) = child.possible_states.remove(&outcome_id) {
                            // 成功找到对应的后续状态节点
                            self.root = next_node;
                            return;
                        }
                    },
                    _ => {
                        // 理论上不会进入这里，除非外部状态同步错误
                    }
                }
                // 如果没找到对应分支（比如之前没探索到），则重置
                self.root = MctsNode::new(1.0, env.get_current_player(), false);
            } else {
                // 确定性节点（移动），直接复用
                self.root = child;
            }
        } else {
            // 树中没有该动作，重置
            self.root = MctsNode::new(1.0, env.get_current_player(), false);
        }
    }

    pub fn run(&mut self, env: &DarkChessEnv) -> Option<usize> {
        let mut total_used = 0;
        
        while total_used < self.config.num_simulations {
            let mut simulation_env = env.clone();
            let (cost, value) = Self::simulate(
                &mut self.root,
                &mut simulation_env,
                None,
                &self.evaluator,
                &self.config,
            );
            
            backpropagate(&mut self.root, value);
            total_used += cost;
            
        }

        self.root.children.iter()
            .max_by_key(|(_, node)| node.visit_count)
            .map(|(action, _)| *action)
    }

    /// 递归模拟
    /// incoming_action: 进入该节点的前置动作（用于 Chance Node 确定位置）
    /// 返回值: (cost, value) - cost 是消耗的评估次数，value 是相对于当前节点行动方的价值
    fn simulate(
        node: &mut MctsNode,
        env: &mut DarkChessEnv,
        incoming_action: Option<usize>,
        evaluator: &Arc<E>,
        config: &MCTSConfig,
    ) -> (usize, f32) {
        let masks = env.action_masks();
        if masks.iter().all(|&x| x == 0) {
            // 游戏结束（无子可走），判负
            return (1, -1.0); 
        }

        // ========================================================================
        // Case A: Chance Node (上一步是翻棋)
        // ========================================================================
        if node.is_chance_node {
            let reveal_pos = incoming_action.expect("Chance node must have incoming action");
            
            // 1. 如果尚未扩展，则进行全量扩展
            if !node.is_expanded {
                // 统计剩余棋子种类和数量
                let mut counts = [0; 6];
                for p in &env.hidden_pieces {
                    counts[get_outcome_id(p)] += 1;
                }
                let total_hidden = env.hidden_pieces.len() as f32;
                
                let mut total_eval_cost = 0;
                let mut total_weighted_value = 0.0;
                
                // 对每一种可能的 outcome 进行扩展和评估
                for outcome_id in 0..6 {
                    if counts[outcome_id] > 0 {
                        let prob = counts[outcome_id] as f32 / total_hidden;
                        
                        // 构造该 outcome 对应的环境
                        let mut next_env = env.clone();
                        let specific_piece = next_env
                            .hidden_pieces
                            .iter()
                            .find(|p| get_outcome_id(p) == outcome_id)
                            .expect("指定类型的棋子不在隐藏池中")
                            .clone();
                        let _ = next_env.step(reveal_pos, Some(specific_piece));
                        
                        let next_player = next_env.get_current_player();
                        let mut child_node = MctsNode::new(1.0, next_player, false);
                        
                        // 递归模拟子节点
                        let (child_cost, child_value) = Self::simulate(
                            &mut child_node,
                            &mut next_env,
                            None,
                            evaluator,
                            config,
                        );
                        
                        total_eval_cost += 1 + child_cost; // 1 for current evaluation + child cost
                        total_weighted_value += prob * child_value; // 机会节点不取反，直接加权平均
                        
                        node.possible_states.insert(outcome_id, (prob, child_node));
                    }
                }
                
                node.is_expanded = true;
                node.visit_count = 1;
                node.value_sum = total_weighted_value;
                
                return (total_eval_cost, total_weighted_value);
            }
            
            // 2. 如果已扩展，则对字典中所有可能的子节点进行MCTS搜索
            let mut total_cost = 0;
            let mut total_weighted_value = 0.0;
            let mut total_visits = 0;
            
            // 对每个可能的 outcome 进行搜索
            for (&outcome_id, (prob, child_node)) in &mut node.possible_states {
                // 构造对应的环境状态
                let mut next_env = env.clone();
                let specific_piece = next_env
                    .hidden_pieces
                    .iter()
                    .find(|p| get_outcome_id(p) == outcome_id)
                    .expect("指定类型的棋子不在隐藏池中")
                    .clone();
                let _ = next_env.step(reveal_pos, Some(specific_piece));
                
                // 递归搜索该子节点
                let (child_cost, child_value) = Self::simulate(
                    child_node,
                    &mut next_env,
                    None,
                    evaluator,
                    config,
                );
                
                total_cost += child_cost;
                total_weighted_value += *prob * child_value;
                total_visits += child_node.visit_count;
            }
            
            // 更新机会节点的统计信息
            node.visit_count = total_visits;
            node.value_sum = total_weighted_value * total_visits as f32;
            
            return (total_cost, total_weighted_value);
        }

        // ========================================================================
        // Case B: State Node (普通节点)
        // ========================================================================
        
        // 1. 扩展 (Expansion)
        if !node.is_expanded {
            let (policy_probs, value) = evaluator.evaluate(env);
            let current_player = env.get_current_player();

            for (action_idx, &mask) in masks.iter().enumerate() {
                if mask == 1 {
                    let prior = policy_probs[action_idx];
                    
                    // 判断该动作是否会导致 Chance Node
                    let is_reveal = action_idx < REVEAL_ACTIONS_COUNT;
                    let child_player = if is_reveal {
                        current_player  // Chance node 保持当前玩家
                    } else {
                        current_player.opposite()  // 移动动作切换玩家
                    };
                    
                    let child_node = MctsNode::new(prior, child_player, is_reveal);
                    node.children.insert(action_idx, child_node);
                }
            }
            node.is_expanded = true;
            return (1, value);
        }

        // 2. 选择 (Selection)
        let (action, best_child) = {
            let sqrt_total_visits = (node.visit_count as f32).sqrt();
            let mut best_action = None;
            let mut best_score = f32::NEG_INFINITY;

            for (&action, child) in &node.children {
                let u_score = config.cpuct * child.prior * sqrt_total_visits / (1.0 + child.visit_count as f32);
                let score = child.q_value() + u_score;
                
                if score > best_score {
                    best_score = score;
                    best_action = Some(action);
                }
            }
            
            let best_action = best_action.expect("No valid child found");
            (best_action, node.children.get_mut(&best_action).unwrap())
        };

        let acting_player = node.player;

        // 3. 执行动作
        match env.step(action, None) {
            Ok((_, _, terminated, truncated, winner_val)) => {
                if terminated || truncated {
                    let value = match winner_val {
                        Some(w) if w == acting_player.val() => 1.0,
                        Some(w) if w == 0 => 0.0,
                        Some(_) => -1.0,
                        None => 0.0,
                    };
                    backpropagate(best_child, value);
                    (0, value)
                } else {
                    // 递归模拟
                    let (cost, child_v) = Self::simulate(best_child, env, Some(action), evaluator, config);
                    
                    // 子节点价值取反（对手视角）
                    let my_value = -child_v;
                    
                    backpropagate(best_child, my_value);
                    (cost, my_value)
                }
            },
            Err(_) => (0, 0.0)
        }
    }
    
    pub fn get_root_probabilities(&self) -> Vec<f32> {
        let mut probs = vec![0.0; ACTION_SPACE_SIZE];
        let total = self.root.visit_count as f32;
        if total == 0.0 { return probs; }
        
        for (&action, child) in &self.root.children {
            if action < probs.len() {
                probs[action] = child.visit_count as f32 / total;
            }
        }
        probs
    }
}

fn backpropagate(node: &mut MctsNode, value: f32) {
    node.visit_count += 1;
    node.value_sum += value;
}