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
use std::fs::OpenOptions;
use std::io::Write;

// ================ CSV日志记录 ================

/// 训练日志记录结构
#[derive(Debug, Clone)]
struct TrainingLog {
    iteration: usize,
    // 损失指标（epoch平均）
    avg_total_loss: f64,
    avg_policy_loss: f64,
    avg_value_loss: f64,
    policy_loss_weight: f64,
    value_loss_weight: f64,
    
    // 场景1: R_A vs B_A
    scenario1_value: f32,
    scenario1_unmasked_a38: f32,
    scenario1_unmasked_a39: f32,
    scenario1_unmasked_a40: f32,
    scenario1_masked_a38: f32,
    scenario1_masked_a39: f32,
    scenario1_masked_a40: f32,
    
    // 场景2: Hidden Threat
    scenario2_value: f32,
    scenario2_unmasked_a3: f32,
    scenario2_unmasked_a5: f32,
    scenario2_masked_a3: f32,
    scenario2_masked_a5: f32,
    
    // 样本统计
    new_samples_count: usize,
    replay_buffer_size: usize,
    avg_game_steps: f32,
    red_win_ratio: f32,
    draw_ratio: f32,
    black_win_ratio: f32,
    avg_policy_entropy: f32,
    high_confidence_ratio: f32,
}

impl TrainingLog {
    fn write_header(csv_path: &str) -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(csv_path)?;
        
        // 检查文件是否为空（新文件需要写入表头）
        let metadata = std::fs::metadata(csv_path)?;
        if metadata.len() == 0 {
            writeln!(file, "iteration,avg_total_loss,avg_policy_loss,avg_value_loss,policy_loss_weight,value_loss_weight,\
                scenario1_value,scenario1_unmasked_a38,scenario1_unmasked_a39,scenario1_unmasked_a40,\
                scenario1_masked_a38,scenario1_masked_a39,scenario1_masked_a40,\
                scenario2_value,scenario2_unmasked_a3,scenario2_unmasked_a5,scenario2_masked_a3,scenario2_masked_a5,\
                new_samples_count,replay_buffer_size,avg_game_steps,red_win_ratio,draw_ratio,black_win_ratio,\
                avg_policy_entropy,high_confidence_ratio")?;
        }
        
        Ok(())
    }
    
    fn append_to_csv(&self, csv_path: &str) -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true)
            .open(csv_path)?;
        
        writeln!(file, "{},{:.6},{:.6},{:.6},{:.3},{:.3},\
            {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},\
            {:.4},{:.4},{:.4},{:.4},{:.4},\
            {},{},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4}",
            self.iteration,
            self.avg_total_loss, self.avg_policy_loss, self.avg_value_loss,
            self.policy_loss_weight, self.value_loss_weight,
            self.scenario1_value, self.scenario1_unmasked_a38, self.scenario1_unmasked_a39, self.scenario1_unmasked_a40,
            self.scenario1_masked_a38, self.scenario1_masked_a39, self.scenario1_masked_a40,
            self.scenario2_value, self.scenario2_unmasked_a3, self.scenario2_unmasked_a5,
            self.scenario2_masked_a3, self.scenario2_masked_a5,
            self.new_samples_count, self.replay_buffer_size, self.avg_game_steps,
            self.red_win_ratio, self.draw_ratio, self.black_win_ratio,
            self.avg_policy_entropy, self.high_confidence_ratio
        )?;
        
        Ok(())
    }
}

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
                if total_batches % 4000 == 0 {
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
        
        // let start_time = Instant::now();
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
            .view([batch_len as i64, 8, 3, 4])  // 禁用状态堆叠后: STATE_STACK_SIZE=1, 所以是8通道
            .to(self.device);
        
        let scalar_tensor = Tensor::from_slice(&scalar_data)
            .view([batch_len as i64, 56])  // 禁用状态堆叠后: 56个特征
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
        
        
        // let elapsed = start_time.elapsed();
        // if batch_len >= 4 {  // 只在批量较大时输出日志
        //     println!("[InferenceServer] 批次处理: {} 个请求耗时 {:.2}ms", 
        //         batch_len, elapsed.as_secs_f64() * 1000.0);
        // }
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

/// 游戏统计信息
#[derive(Debug, Clone)]
struct GameStats {
    steps: usize,
    winner: Option<i32>,  // Some(1)=红胜, Some(-1)=黑胜, None/Some(0)=平局
}

/// 单局游戏的完整数据（包含样本和元数据）
#[derive(Debug, Clone)]
struct GameEpisode {
    samples: Vec<(Observation, Vec<f32>, f32, Vec<i32>)>,
    game_length: usize,
    winner: Option<i32>,
}

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

    /// 运行一局自对弈游戏，返回GameEpisode
    pub fn play_episode(&self, episode_num: usize) -> GameEpisode {
        println!("  [Worker-{}] 开始第 {} 局游戏", self.worker_id, episode_num + 1);
        let start_time = Instant::now();
        
        let mut env = DarkChessEnv::new();
        let config = MCTSConfig { num_simulations: self.mcts_sims, cpuct: 1.0 };
        let mut mcts = MCTS::new(&env, self.evaluator.clone(), config);
        
        let mut episode_data = Vec::new();
        let mut step = 0;
        
        // 🐛 DEBUG: 记录首步MCTS详情
        let debug_first_step = episode_num < 2; // 只调试前2局
        
        loop {
            // 运行MCTS
            mcts.run();
            let probs = mcts.get_root_probabilities();
            let masks = env.action_masks();
            
            // 🐛 DEBUG: 打印MCTS根节点详情
            if debug_first_step && step < 3 {
                println!("    [Worker-{}] Step {}: MCTS根节点详情", self.worker_id, step);
                let top_actions = get_top_k_actions(&probs, 5);
                for (action, prob) in top_actions {
                    println!("      action={}, prob={:.3}", action, prob);
                }
            }
            
            // 保存数据
            episode_data.push((
                env.get_state(),
                probs.clone(),
                env.get_current_player(),
                masks,
            ));
            
            // 选择动作(使用更长的高温探索期,并提高探索温度)
            // 游戏平均步数在13步左右
            let temperature = if step < 2 { 1.5 } else if step < 10 { 1.2 } else { 0.9 };
            let action = sample_action(&probs, &env, temperature);
            
            // 🐛 DEBUG: 记录动作选择
            if debug_first_step && step < 3 {
                println!("      选择: action={}, temp={:.1}", action, temperature);
            }
            
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
                        
                        // 🐛 DEBUG: 检查价值标签分布
                        if debug_first_step {
                            let mut red_values = Vec::new();
                            let mut black_values = Vec::new();
                            for (_, _, player, _) in &episode_data {
                                let val = if player.val() == 1 { reward_red } else { -reward_red };
                                if player.val() == 1 {
                                    red_values.push(val);
                                } else {
                                    black_values.push(val);
                                }
                            }
                            println!("    [Worker-{}] 价值标签统计: 红方样本数={}, 黑方样本数={}", 
                                self.worker_id, red_values.len(), black_values.len());
                            if !red_values.is_empty() {
                                println!("      红方价值标签: {:.2} (winner={:?})", red_values[0], winner);
                            }
                            if !black_values.is_empty() {
                                println!("      黑方价值标签: {:.2} (winner={:?})", black_values[0], winner);
                            }
                        }
                        
                        // 回填价值
                        let mut samples = Vec::new();
                        for (obs, p, player, mask) in episode_data {
                            let val = if player.val() == 1 { reward_red } else { -reward_red };
                            samples.push((obs, p, val, mask));
                        }
                        
                        return GameEpisode {
                            samples,
                            game_length: step,
                            winner,
                        };
                    }
                },
                Err(e) => {
                    eprintln!("[Worker-{}] 游戏错误: {}", self.worker_id, e);
                    return GameEpisode {
                        samples: Vec::new(),
                        game_length: step,
                        winner: None,
                    };
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
                return GameEpisode {
                    samples,
                    game_length: step,
                    winner: None,
                };
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

/// 🐛 DEBUG: 获取top-k动作
fn get_top_k_actions(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(k).collect()
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
            game_length INTEGER NOT NULL,
            step_in_game INTEGER NOT NULL,
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
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_game_length ON training_samples(game_length)",
        [],
    )?;
    
    println!("数据库初始化完成: {}", db_path);
    Ok(conn)
}

fn save_samples_to_db(
    conn: &mut Connection,
    iteration: usize,
    episode_type: &str,
    samples: &[(Observation, Vec<f32>, f32, Vec<i32>)],
    game_length: usize,
) -> Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO training_samples 
             (iteration, episode_type, board_state, scalar_state, policy_probs, value_target, action_mask, game_length, step_in_game) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
        )?;
        
        for (step_idx, (obs, probs, value, mask)) in samples.iter().enumerate() {
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
                game_length as i64,
                step_idx as i64,
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
    
    // 动态调整策略权重: 早期更注重策略学习,后期平衡
    let policy_weight = 1.5 + (epoch as f32 * 0.1).min(1.0); // 从1.5逐渐增加到2.5
    let value_weight = 2.0; // 大幅提高价值权重 (原来是0.5-1.0隐式权重)
    
    // 🐛 DEBUG: 检查样本统计
    let mut value_stats = Vec::new();
    let mut entropy_stats = Vec::new();
    
    for batch_start in (0..shuffled_examples.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(shuffled_examples.len());
        let batch = &shuffled_examples[batch_start..batch_end];
        
        for (obs, target_probs, target_val, masks) in batch.iter() {
            // 🐛 DEBUG: 收集统计数据
            value_stats.push(*target_val);
            let entropy: f32 = target_probs.iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| -p * p.ln())
                .sum();
            entropy_stats.push(entropy);
            
            let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap()).view([1, 8, 3, 4]).to(device);
            let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap()).view([1, 56]).to(device);
            let target_p = Tensor::from_slice(target_probs).view([1, 46]).to(device);
            let target_v = Tensor::from_slice(&[*target_val]).view([1, 1]).to(device);
            
            let mask_vec: Vec<f32> = masks.iter().map(|&m| m as f32).collect();
            let mask_tensor = Tensor::from_slice(&mask_vec).view([1, 46]).to(device);
            
            let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
            
            let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
            let log_probs = masked_logits.log_softmax(-1, Kind::Float);
            
            // 策略损失: 交叉熵
            let p_loss = (&target_p * &log_probs).sum(Kind::Float).neg() * (policy_weight as f64);
            // 价值损失: MSE,加大权重
            let v_loss = value.mse_loss(&target_v, tch::Reduction::Mean) * (value_weight as f64);
            
            let total_loss = &p_loss + &v_loss;
            
            opt.backward_step(&total_loss);
            
            total_loss_sum += total_loss.double_value(&[]);
            policy_loss_sum += p_loss.double_value(&[]) / policy_weight as f64;
            value_loss_sum += v_loss.double_value(&[]) / value_weight as f64;
            num_samples += 1;
        }
    }
    
    // 🐛 DEBUG: 输出样本质量统计
    if epoch == 0 && !value_stats.is_empty() {
        let avg_value: f32 = value_stats.iter().sum::<f32>() / value_stats.len() as f32;
        let std_value: f32 = (value_stats.iter().map(|v| (v - avg_value).powi(2)).sum::<f32>() / value_stats.len() as f32).sqrt();
        let avg_entropy: f32 = entropy_stats.iter().sum::<f32>() / entropy_stats.len() as f32;
        
        let positive_values = value_stats.iter().filter(|&&v| v > 0.0).count();
        let negative_values = value_stats.iter().filter(|&&v| v < 0.0).count();
        let zero_values = value_stats.iter().filter(|&&v| v == 0.0).count();
        
        println!("    🐛 样本统计: 总数={}, 价值[avg={:.3}, std={:.3}], 熵[avg={:.3}]", 
            value_stats.len(), avg_value, std_value, avg_entropy);
        println!("    🐛 价值分布: 正={} ({:.1}%), 零={} ({:.1}%), 负={} ({:.1}%)",
            positive_values, positive_values as f32 / value_stats.len() as f32 * 100.0,
            zero_values, zero_values as f32 / value_stats.len() as f32 * 100.0,
            negative_values, negative_values as f32 / value_stats.len() as f32 * 100.0);
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

/// 场景验证结果
#[derive(Debug, Clone)]
struct ScenarioResult {
    value: f32,
    unmasked_probs: Vec<f32>,  // 原始softmax概率
    masked_probs: Vec<f32>,    // 应用mask后的概率
}

/// 验证模型在标准场景上的表现，返回详细数据
fn validate_model_on_scenarios(vs: &nn::VarStore, device: Device, _iteration: usize) -> (ScenarioResult, ScenarioResult) {
    use banqi_3x4::game_env::Player;
    
    let net = BanqiNet::new(&vs.root());
    
    // 场景1: R_A vs B_A
    let scenario1_result = {
        let mut env = DarkChessEnv::new();
        env.setup_two_advisors(Player::Black);
        
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 8, 3, 4])
            .to(device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 56])
            .to(device);
        
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(device).view([1, 46]);
        
        let (logits, value) = tch::no_grad(|| net.forward(&board_tensor, &scalar_tensor));
        
        // 🐛 DEBUG: 打印原始logits
        let logits_vec: Vec<f32> = (0..46).map(|i| logits.double_value(&[0, i]) as f32).collect();
        let top_logits = get_top_k_actions(&logits_vec, 5);
        println!("      🐛 原始logits (top-5): {:?}", top_logits);
        
        // 未应用mask的概率分布
        let unmasked_probs_tensor = logits.softmax(-1, Kind::Float);
        let unmasked_probs: Vec<f32> = (0..46).map(|i| unmasked_probs_tensor.double_value(&[0, i]) as f32).collect();
        
        // 应用mask后的概率分布
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let masked_probs_tensor = masked_logits.softmax(-1, Kind::Float);
        let masked_probs: Vec<f32> = (0..46).map(|i| masked_probs_tensor.double_value(&[0, i]) as f32).collect();
        
        let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
        
        // 🐛 DEBUG: 检查有效动作
        let valid_actions: Vec<usize> = masks.iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1.0 { Some(i) } else { None })
            .collect();
        println!("      🐛 有效动作数: {}, 包括: {:?}", valid_actions.len(), &valid_actions[..valid_actions.len().min(10)]);
        
        println!("    场景1 (R_A vs B_A): value={:.3}", value_pred);
        println!("      未应用mask: a38={:.1}%, a39={:.1}%, a40={:.1}%", 
            unmasked_probs[38]*100.0, unmasked_probs[39]*100.0, unmasked_probs[40]*100.0);
        println!("      应用mask后: a38={:.1}%, a39={:.1}%, a40={:.1}%", 
            masked_probs[38]*100.0, masked_probs[39]*100.0, masked_probs[40]*100.0);
        println!("      期望: action38主导(>90%), value应偏向当前玩家(黑方)略优或平局");
        
        ScenarioResult {
            value: value_pred,
            unmasked_probs,
            masked_probs,
        }
    };
    
    // 场景2: Hidden Threat
    let scenario2_result = {
        let mut env = DarkChessEnv::new();
        env.setup_hidden_threats();
        
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 8, 3, 4])
            .to(device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 56])
            .to(device);
        
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(device).view([1, 46]);
        
        let (logits, value) = tch::no_grad(|| net.forward(&board_tensor, &scalar_tensor));
        
        // 🐛 DEBUG: 打印原始logits
        let logits_vec: Vec<f32> = (0..46).map(|i| logits.double_value(&[0, i]) as f32).collect();
        let top_logits = get_top_k_actions(&logits_vec, 5);
        println!("      🐛 原始logits (top-5): {:?}", top_logits);
        
        // 未应用mask的概率分布
        let unmasked_probs_tensor = logits.softmax(-1, Kind::Float);
        let unmasked_probs: Vec<f32> = (0..46).map(|i| unmasked_probs_tensor.double_value(&[0, i]) as f32).collect();
        
        // 应用mask后的概率分布
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let masked_probs_tensor = masked_logits.softmax(-1, Kind::Float);
        let masked_probs: Vec<f32> = (0..46).map(|i| masked_probs_tensor.double_value(&[0, i]) as f32).collect();
        
        let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
        
        // 🐛 DEBUG: 检查有效动作
        let valid_actions: Vec<usize> = masks.iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1.0 { Some(i) } else { None })
            .collect();
        println!("      🐛 有效动作数: {}, 包括: {:?}", valid_actions.len(), &valid_actions[..valid_actions.len().min(10)]);
        
        println!("    场景2 (Hidden Threat): value={:.3}", value_pred);
        println!("      未应用mask: a3={:.1}%, a5={:.1}%", 
            unmasked_probs[3]*100.0, unmasked_probs[5]*100.0);
        println!("      应用mask后: a3={:.1}%, a5={:.1}%", 
            masked_probs[3]*100.0, masked_probs[5]*100.0);
        println!("      期望: action3主导(>90%), value应能反映位置优势");
        
        ScenarioResult {
            value: value_pred,
            unmasked_probs,
            masked_probs,
        }
    };
    
    (scenario1_result, scenario2_result)
}

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
    let num_workers = (num_cpus::get() * 2).max(8); // 工作线程数:CPU核心数的2倍,至少8个
    let mcts_sims = 1200; // 进一步提高MCTS质量 - 这是训练数据质量的关键
    let num_episodes_per_iteration = 80; // 增加游戏数以收集更多样本
    let inference_batch_size = 64.min(num_workers); // 推理批量大小
    let inference_timeout_ms = 5; // 批量推理超时(毫秒)- 进一步降低以提高响应速度
    let max_buffer_size = 25000; // 经验回放缓冲区 - 保留最近25000个样本
    
    println!("\n=== 并行训练配置 ===");
    println!("工作线程数: {}", num_workers);
    println!("每轮游戏数: {}", num_episodes_per_iteration);
    println!("MCTS模拟次数: {}", mcts_sims);
    println!("推理批量大小: {}", inference_batch_size);
    println!("推理超时: {}ms", inference_timeout_ms);
    println!("经验回放缓冲区: {}", max_buffer_size);
    
    // 初始化数据库
    let db_path = "training_samples.db";
    let mut conn = init_database(db_path)?;
    
    // 创建模型和优化器
    let vs = nn::VarStore::new(device);
    // **重要**: 立即创建网络以初始化所有参数
    let _init_net = BanqiNet::new(&vs.root());
    
    let learning_rate = 2e-4; // 降低学习率避免震荡 (从5e-4降到2e-4)
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;
    
    // 训练超参数
    let num_iterations = 200;
    let batch_size = 128; // 增大批量以稳定训练
    let epochs_per_iteration = 5; // 大幅减少epoch避免过拟合 (从15降到5)
    
    // 第一阶段：加载已有数据训练
    println!("\n=== 第一阶段：加载已有数据 ===");
    let existing_samples = load_samples_from_db(&conn)?;
    if !existing_samples.is_empty() {
        println!("加载了 {} 个样本", existing_samples.len());
        
        // 创建一个临时网络用于初始训练
        let temp_net = BanqiNet::new(&vs.root());
        
        for epoch in 0..5 {
            let (loss, p_loss, v_loss) = train_step(&mut opt, &temp_net, &existing_samples, batch_size, device, epoch);

                println!("  Epoch {}/5, Loss={:.4} (Policy={:.4}, Value={:.4})", 
                    epoch + 1, loss, p_loss, v_loss);
            
        }
        
        vs.save("banqi_model_pretrained.ot")?;
        println!("已保存预训练模型");
    }
    
    // 第二阶段：并行自对弈训练
    println!("\n=== 第二阶段：并行自对弈训练 ===");
    
    // 初始化CSV日志
    let csv_path = "training_log.csv";
    TrainingLog::write_header(csv_path)?;
    println!("CSV日志文件: {}", csv_path);
    
    // 经验回放缓冲区
    let mut replay_buffer: Vec<(Observation, Vec<f32>, f32, Vec<i32>)> = Vec::new();
    
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
                
                let mut all_episodes = Vec::new();
                let episodes_per_worker = (num_episodes_per_iteration + num_workers - 1) / num_workers;
                
                for ep in 0..episodes_per_worker {
                    let episode = worker.play_episode(ep);
                    all_episodes.push(episode);
                }
                
                println!("  [Worker-{}] 完成所有 {} 局游戏", worker_id, episodes_per_worker);
                result_tx.send(all_episodes).expect("无法发送结果");
            });
            
            worker_handles.push(handle);
        }
        
        // 关闭主请求发送端，以便推理服务器知道何时退出
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
        
        // 从episodes中提取统计信息和样本
        let mut all_samples = Vec::new();
        let mut all_game_stats = Vec::new();
        for episode in &all_episodes {
            all_samples.extend(episode.samples.clone());
            all_game_stats.push(GameStats {
                steps: episode.game_length,
                winner: episode.winner,
            });
        }
        
        println!("  收集了 {} 个训练样本（来自 {} 局游戏）", all_samples.len(), all_episodes.len());
        
        // 计算游戏统计信息
        let total_games = all_game_stats.len();
        let total_steps: usize = all_game_stats.iter().map(|s| s.steps).sum();
        let avg_game_steps = if total_games > 0 { total_steps as f32 / total_games as f32 } else { 0.0 };
        
        let mut red_wins = 0;
        let mut black_wins = 0;
        let mut draws = 0;
        for stat in &all_game_stats {
            match stat.winner {
                Some(1) => red_wins += 1,
                Some(-1) => black_wins += 1,
                _ => draws += 1,
            }
        }
        
        let red_win_ratio = if total_games > 0 { red_wins as f32 / total_games as f32 } else { 0.0 };
        let black_win_ratio = if total_games > 0 { black_wins as f32 / total_games as f32 } else { 0.0 };
        let draw_ratio = if total_games > 0 { draws as f32 / total_games as f32 } else { 0.0 };
        
        // 计算策略熵和高置信度样本比例
        let mut total_entropy = 0.0f32;
        let mut high_confidence_count = 0;
        
        // 🐛 DEBUG: 收集策略分布统计
        let mut max_probs = Vec::new();
        let mut action_diversity = Vec::new();
        
        for (_, probs, _, _) in &all_samples {
            let entropy: f32 = probs.iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| -p * p.ln())
                .sum();
            total_entropy += entropy;
            if entropy < 1.5 {
                high_confidence_count += 1;
            }
            
            // 🐛 统计最大概率和有效动作数
            let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
            max_probs.push(max_prob);
            let num_significant_actions = probs.iter().filter(|&&p| p > 0.01).count();
            action_diversity.push(num_significant_actions);
        }
        
        let avg_policy_entropy = if !all_samples.is_empty() { 
            total_entropy / all_samples.len() as f32 
        } else { 
            0.0 
        };
        let high_confidence_ratio = if !all_samples.is_empty() {
            high_confidence_count as f32 / all_samples.len() as f32
        } else {
            0.0
        };
        
        // 数据质量诊断
        if iteration % 10 == 0 {
            println!("  ========== 数据质量诊断 ==========");
            println!("    游戏统计: 总局数={}, 平均步数={:.1}", total_games, avg_game_steps);
            println!("    游戏结果: 红胜={} ({:.1}%), 平局={} ({:.1}%), 黑胜={} ({:.1}%)", 
                red_wins, red_win_ratio * 100.0,
                draws, draw_ratio * 100.0,
                black_wins, black_win_ratio * 100.0);
            println!("    策略质量: 平均熵={:.3}, 高置信度样本={} ({:.1}%)", 
                avg_policy_entropy, high_confidence_count, high_confidence_ratio * 100.0);
            
            // 🐛 DEBUG: 输出策略分布质量
            if !max_probs.is_empty() {
                let avg_max_prob: f32 = max_probs.iter().sum::<f32>() / max_probs.len() as f32;
                let avg_diversity: f32 = action_diversity.iter().map(|&x| x as f32).sum::<f32>() / action_diversity.len() as f32;
                println!("    🐛 策略分布: 平均最大概率={:.3}, 平均有效动作数={:.1}", avg_max_prob, avg_diversity);
                
                // 统计完全均匀分布的样本（可能表示MCTS未收敛）
                let uniform_samples = max_probs.iter().filter(|&&p| p < 0.1).count();
                println!("    🐛 异常样本: 近似均匀分布={} ({:.1}%)", 
                    uniform_samples, uniform_samples as f32 / max_probs.len() as f32 * 100.0);
            }
        }
        
        // 保存样本到数据库（按episode分别保存，带游戏长度信息）
        for episode in &all_episodes {
            save_samples_to_db(&mut conn, iteration, "self_play", &episode.samples, episode.game_length)?;
        }
        
        // 保存新样本数量（在移动all_samples之前）
        let new_samples_count = all_samples.len();
        
        // 更新经验回放缓冲区
        replay_buffer.extend(all_samples);
        if replay_buffer.len() > max_buffer_size {
            // 保留最新的样本
            let remove_count = replay_buffer.len() - max_buffer_size;
            replay_buffer.drain(0..remove_count);
        }
        println!("  经验回放缓冲区: {} 个样本", replay_buffer.len());
        
        println!("  开始训练...");
        
        // 获取当前训练epoch的策略和价值损失权重
        let policy_weight = 1.5 + (0 as f32 * 0.1).min(1.0); // 从train_step获取 - 这里取第一个epoch的值
        let value_weight = 2.0;
        
        // 训练模型 - 使用经验回放缓冲区而非仅当前样本
        let temp_net = BanqiNet::new(&vs.root());
        let mut total_losses = Vec::new();
        let mut policy_losses = Vec::new();
        let mut value_losses = Vec::new();
        
        let train_start = Instant::now();
        for epoch in 0..epochs_per_iteration {
            let (loss, p_loss, v_loss) = train_step(&mut opt, &temp_net, &replay_buffer, batch_size, device, epoch);
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
        println!("  训练完成,耗时 {:.1}s,平均Loss: {:.4} (Policy={:.4}, Value={:.4})", 
            train_elapsed.as_secs_f64(), avg_loss, avg_p_loss, avg_v_loss);
        
        // 验证模型性能并收集场景数据
        println!("\n  ========== 模型验证 (Iteration {}) ==========", iteration);
        let (scenario1, scenario2) = validate_model_on_scenarios(&vs, device, iteration);
        
        // 构建训练日志
        let log = TrainingLog {
            iteration,
            avg_total_loss: avg_loss,
            avg_policy_loss: avg_p_loss,
            avg_value_loss: avg_v_loss,
            policy_loss_weight: policy_weight as f64,
            value_loss_weight: value_weight as f64,
            
            scenario1_value: scenario1.value,
            scenario1_unmasked_a38: scenario1.unmasked_probs[38],
            scenario1_unmasked_a39: scenario1.unmasked_probs[39],
            scenario1_unmasked_a40: scenario1.unmasked_probs[40],
            scenario1_masked_a38: scenario1.masked_probs[38],
            scenario1_masked_a39: scenario1.masked_probs[39],
            scenario1_masked_a40: scenario1.masked_probs[40],
            
            scenario2_value: scenario2.value,
            scenario2_unmasked_a3: scenario2.unmasked_probs[3],
            scenario2_unmasked_a5: scenario2.unmasked_probs[5],
            scenario2_masked_a3: scenario2.masked_probs[3],
            scenario2_masked_a5: scenario2.masked_probs[5],
            
            new_samples_count,
            replay_buffer_size: replay_buffer.len(),
            avg_game_steps,
            red_win_ratio,
            draw_ratio,
            black_win_ratio,
            avg_policy_entropy,
            high_confidence_ratio,
        };
        
        // 写入CSV
        if let Err(e) = log.append_to_csv(csv_path) {
            eprintln!("  警告: 无法写入CSV日志: {}", e);
        } else {
            println!("  已写入训练日志到 {}", csv_path);
        }
        
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
