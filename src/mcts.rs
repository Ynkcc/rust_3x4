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
        let mut budget = self.config.num_simulations;
        
        while budget > 0 {
            let mut simulation_env = env.clone();
            // simulate 返回本次模拟消耗的评估次数
            let cost = Self::simulate(
                &mut self.root,
                &mut simulation_env,
                None,
                &self.evaluator,
                &self.config,
            );
            
            if budget >= cost {
                budget -= cost;
            } else {
                break;
            }
        }

        self.root.children.iter()
            .max_by_key(|(_, node)| node.visit_count)
            .map(|(action, _)| *action)
    }

    /// 递归模拟
    /// incoming_action: 进入该节点的前置动作（用于 Chance Node 确定位置）
    fn simulate(
        node: &mut MctsNode,
        env: &mut DarkChessEnv,
        incoming_action: Option<usize>,
        evaluator: &Arc<E>,
        config: &MCTSConfig,
    ) -> usize {
        let masks = env.action_masks();
        if masks.iter().all(|&x| x == 0) {
            // 游戏结束
            return 1; 
        }

        // ========================================================================
        // Case A: Chance Node (上一步是翻棋)
        // ========================================================================
        if node.is_chance_node {
            let reveal_pos = incoming_action.expect("Chance node must have incoming action");
            
            // 1. 如果尚未扩展（即 possible_states 为空），则进行全量扩展
            if !node.is_expanded {
                // 获取当前剩余隐藏棋子的概率分布
                // 为了精确计算，我们需要统计 hidden_pieces
                // 注意：env.hidden_pieces 是 Bag，我们通过 clone 不同状态来模拟
                
                // 统计剩余棋子种类和数量
                let mut counts = [0; 6];
                for p in &env.hidden_pieces {
                    counts[get_outcome_id(p)] += 1;
                }
                let total_hidden = env.hidden_pieces.len() as f32;
                
                let mut eval_cost = 0;
                
                // 对每一种可能的 outcome 进行扩展和评估
                for outcome_id in 0..6 {
                    if counts[outcome_id] > 0 {
                        let prob = counts[outcome_id] as f32 / total_hidden;
                        
                        // 构造该 outcome 对应的环境
                        // 方法：克隆当前环境，强制翻出指定棋子
                        // 注意：这里需要 hack 一下 env，或者利用 env 提供的机制
                        // 我们在 Env 中有 hidden_pieces，我们需要从中找到该类型的棋子并放到 reveal_pos
                        
                        let mut next_env = env.clone();
                        // 找到该类型棋子，克隆出来，交给 step 指定翻出
                        let specific_piece = next_env
                            .hidden_pieces
                            .iter()
                            .find(|p| get_outcome_id(p) == outcome_id)
                            .expect("指定类型的棋子不在隐藏池中")
                            .clone();
                        // 调用 step，强制翻出 specific_piece
                        let _ = next_env.step(reveal_pos, Some(specific_piece));
                        
                        let (_policy, value_est) = evaluator.evaluate(&next_env); // 这里的 value_est 是 V(s')
                        
                        // 创建子节点 (State Node)
                        let next_player = next_env.get_current_player();
                        let mut child_node = MctsNode::new(1.0, next_player, false); // Prior 暂设 1.0 或基于网络
                        child_node.value_sum = value_est; // 初始化价值
                        child_node.visit_count = 1;
                        
                        // 存入 possible_states
                        node.possible_states.insert(outcome_id, (prob, child_node));
                        eval_cost += 1;
                    }
                }
                node.is_expanded = true;
                
                // 计算 Chance Node 的价值 (加权平均)
                let mut weighted_value = 0.0;
                for (_, (prob, child)) in &node.possible_states {
                    // 子节点价值是相对于子节点玩家的，对当前节点来说需要取反
                    weighted_value += prob * (-child.q_value());
                }
                
                // 更新当前 Chance Node
                // Chance Node 的 visit_count = sum(children visits)
                node.visit_count = node.possible_states.values().map(|(_, c)| c.visit_count).sum();
                node.value_sum = weighted_value * node.visit_count as f32; // 反推 Sum
                
                return eval_cost;
            }
            
            // 2. 如果已扩展，则根据环境实际情况选择分支深入
            // 这里我们必须依据传入的 env 的真实情况（这是模拟过程中的真实）
            // 但是！传入的 env 是从 Root Clone 下来的。在到达 Chance Node 之前，我们并没有执行 Reveal。
            // Wait: 流程是 Select -> Step -> Simulate.
            // 所以传入的 env 已经执行过 Step(Reveal) 了。
            // 它是确定性的某一种 outcome。
            
            // 检查 env 中 reveal_pos 的棋子
            let slot = &env.get_board_slots()[reveal_pos];
            let outcome_id = match slot {
                Slot::Revealed(p) => get_outcome_id(p),
                _ => panic!("Chance node logic error: expected revealed piece at {}", reveal_pos),
            };
            
            if let Some((_prob, next_node)) = node.possible_states.get_mut(&outcome_id) {
                // 递归深入
                let cost = Self::simulate(next_node, env, None, evaluator, config);
                
                // 回溯：更新 Chance Node
                // 重新计算加权价值
                let mut weighted_val = 0.0;
                let mut total_visits = 0;
                for (_, (p, c)) in &node.possible_states {
                    weighted_val += p * (-c.q_value()); // 子节点价值取反
                    total_visits += c.visit_count;
                }
                node.visit_count = total_visits;
                node.value_sum = weighted_val * total_visits as f32;
                
                return cost;
            } else {
                // 理论上不可能，除非概率极小未被采样（全量扩展不会发生）
                return 0;
            }
        }

        // ========================================================================
        // Case B: State Node (普通节点)
        // ========================================================================
        
        // 1. 扩展 (Expansion)
        if !node.is_expanded {
            let (policy_probs, _value) = evaluator.evaluate(env);
            
            let current_player = env.get_current_player();

            for (action_idx, &mask) in masks.iter().enumerate() {
                if mask == 1 {
                    let prior = policy_probs[action_idx];
                    
                    // 判断该动作是否会导致 Chance Node
                    let is_reveal = action_idx < REVEAL_ACTIONS_COUNT;
                    
                    // 注意：Chance Node 的 Player 依然是当前行动方? 
                    // 不，Chance Node 代表“环境正在行动”。
                    // 但为了简化 value 取反逻辑，通常 Chance Node 归属权保持不变，或者视为中立。
                    // 这里我们设 Chance Node 的 Player 为 current_player，
                    // 这样它的子节点（Outcome）的价值（相对于 next_player）取反后就是 Chance Node 的价值。
                    let child_node = MctsNode::new(prior, current_player, is_reveal);
                    node.children.insert(action_idx, child_node);
                }
            }
            node.is_expanded = true;
            return 1; // 消耗 1 次评估
        }

        // 2. 选择 (Selection)
    // Select child action first (immutable borrow), then get mutable child
    let (action, _) = select_child(node, config.cpuct);
    let acting_player = node.player; // copy out before mutable borrow
    let best_child = node.children.get_mut(&action).expect("Child must exist after selection");

        // 3. 执行动作
        // 注意：如果 action 是 reveal，step 会随机翻。
        // 对于 State Node 的 simulate，我们只负责往下走。
        // 如果 best_child 是 Chance Node，我们进入 Chance 逻辑。
        // 此时 env.step(action) 会产生一个随机 outcome。
        // 下一层的 simulate(chance_node) 会读取这个 outcome 并导向正确的 state node。
        
        match env.step(action, None) {
            Ok((_, _, terminated, truncated, winner_val)) => {
                let value;
                if terminated || truncated {
                    if let Some(w) = winner_val {
                        if w == acting_player.val() {
                            value = 1.0;
                        } else if w == 0 {
                            value = 0.0;
                        } else {
                            value = -1.0;
                        }
                    } else {
                        value = 0.0;
                    }
                    backpropagate(best_child, value);
                    return 0; // 终端节点不消耗评估（或者算0）
                } else {
                    // 递归
                    // 注意：这里传入 action，以便 Chance Node 知道查哪里
                    let cost = Self::simulate(best_child, env, Some(action), evaluator, config);
                    
                    // 获取子节点现在的 Q 值
                    // 如果子节点是 Chance Node，它的 Q 值已经是加权平均后的
                    // 如果子节点是 State Node，它的 Q 值是常规的
                    // 无论如何，我们取反
                    value = -best_child.q_value();
                    
                    backpropagate(best_child, value);
                    return cost;
                }
            },
            Err(_) => return 0
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

fn select_child<'a>(node: &'a mut MctsNode, cpuct: f32) -> (usize, &'a mut MctsNode) {
    let sqrt_total_visits = (node.visit_count as f32).sqrt();

    let best_action = node.children.iter()
        .max_by(|(_, node_a), (_, node_b)| {
            let u_a = node_a.q_value() + cpuct * node_a.prior * sqrt_total_visits / (1.0 + node_a.visit_count as f32);
            let u_b = node_b.q_value() + cpuct * node_b.prior * sqrt_total_visits / (1.0 + node_b.visit_count as f32);
            u_a.partial_cmp(&u_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(a, _)| *a)
        .expect("Error selecting child");

    (best_action, node.children.get_mut(&best_action).unwrap())
}

fn backpropagate(node: &mut MctsNode, value: f32) {
    node.visit_count += 1;
    node.value_sum += value;
}