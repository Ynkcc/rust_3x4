// code_files/src/tauri_main.rs
use banqi_3x4::*;
use banqi_3x4::ai::{Policy, RandomPolicy, RevealFirstPolicy};
use serde::{Serialize, Deserialize}; // Added Deserialize
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{Manager, State};

// 游戏状态的可序列化版本
#[derive(Debug, Clone, Serialize)]
struct GameState {
    board: Vec<String>,
    current_player: String,
    move_counter: usize,
    total_step_counter: usize,
    dead_red: Vec<String>,
    dead_black: Vec<String>,
    action_masks: Vec<i32>,
    reveal_probabilities: Vec<f32>,
    bitboards: HashMap<String, Vec<bool>>,
}

#[derive(Debug, Clone, Serialize)]
struct StepResult {
    state: GameState,
    terminated: bool,
    truncated: bool,
    winner: Option<i32>,
}

// 对手类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OpponentType {
    PvP,         // 本地双人
    Random,      // 随机对手
    RevealFirst, // 优先翻棋
}

// 应用状态：包含游戏环境和当前对手设置
struct AppState {
    game: Mutex<DarkChessEnv>,
    opponent_type: Mutex<OpponentType>,
}

// Tauri 命令：重置游戏
#[tauri::command]
fn reset_game(opponent: Option<String>, state: State<AppState>) -> GameState {
    let mut game = state.game.lock().unwrap();
    let mut opp_type_lock = state.opponent_type.lock().unwrap();

    // 设置对手类型
    *opp_type_lock = match opponent.as_deref() {
        Some("Random") => OpponentType::Random,
        Some("RevealFirst") => OpponentType::RevealFirst,
        _ => OpponentType::PvP,
    };

    game.reset();
    extract_game_state(&*game)
}

// Tauri 命令：执行动作
#[tauri::command]
fn step_game(action: usize, state: State<AppState>) -> Result<StepResult, String> {
    let mut game = state.game.lock().unwrap();
    
    match game.step(action) {
        Ok((_obs, _reward, terminated, truncated, winner)) => {
            let state_data = extract_game_state(&*game);
            Ok(StepResult {
                state: state_data,
                terminated,
                truncated,
                winner,
            })
        }
        Err(e) => Err(e),
    }
}

// Tauri 命令：执行 AI 动作
#[tauri::command]
fn bot_move(state: State<AppState>) -> Result<StepResult, String> {
    let mut game = state.game.lock().unwrap();
    let opp_type = *state.opponent_type.lock().unwrap();
    
    // 如果处于 PvP，提示前端无需调用 AI
    if opp_type == OpponentType::PvP {
        return Err("当前为本地双人模式，无需 AI 行动".to_string());
    }

    // 调用策略模块选择动作
    let chosen_action = match opp_type {
        OpponentType::RevealFirst => RevealFirstPolicy::choose_action(&*game),
        OpponentType::Random => RandomPolicy::choose_action(&*game),
        OpponentType::PvP => None, // 已在上面返回 Err，这里兜底
    }.ok_or_else(|| "AI 无棋可走".to_string())?;

    match game.step(chosen_action) {
        Ok((_obs, _reward, terminated, truncated, winner)) => {
            let state_data = extract_game_state(&*game);
            Ok(StepResult {
                state: state_data,
                terminated,
                truncated,
                winner,
            })
        }
        Err(e) => Err(e),
    }
}

// Tauri 命令：获取当前状态
#[tauri::command]
fn get_game_state(state: State<AppState>) -> GameState {
    let game = state.game.lock().unwrap();
    extract_game_state(&*game)
}

// Tauri 命令：获取对手类型
#[tauri::command]
fn get_opponent_type(state: State<AppState>) -> OpponentType {
    *state.opponent_type.lock().unwrap()
}

// Tauri 命令：获取移动动作编号
#[tauri::command]
fn get_move_action(from_sq: usize, to_sq: usize, state: State<AppState>) -> Option<usize> {
    let game = state.game.lock().unwrap();
    game.get_action_for_coords(&vec![from_sq, to_sq])
}

// 辅助函数：获取棋子短名称
fn get_piece_short_name(piece: &Piece) -> String {
    let p_char = match piece.player {
        Player::Red => "R",
        Player::Black => "B",
    };
    let t_char = match piece.piece_type {
        PieceType::General => "Gen",
        PieceType::Advisor => "Adv",
        PieceType::Soldier => "Sol",
    };
    format!("{}_{}", p_char, t_char)
}

// 辅助函数：从游戏环境中提取状态
fn extract_game_state(env: &DarkChessEnv) -> GameState {
    let board_slots = env.get_board_slots();
    let board: Vec<String> = board_slots
        .iter()
        .map(|slot| match slot {
            Slot::Empty => "Empty".to_string(),
            Slot::Hidden => "Hidden".to_string(),
            Slot::Revealed(piece) => get_piece_short_name(piece),
        })
        .collect();
    
    let current_player = match env.get_current_player() {
        Player::Red => "Red".to_string(),
        Player::Black => "Black".to_string(),
    };
    
    let dead_red: Vec<String> = env
        .get_dead_pieces(Player::Red)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();
    
    let dead_black: Vec<String> = env
        .get_dead_pieces(Player::Black)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();
    
    let action_masks = env.action_masks();
    let reveal_probabilities = env.get_reveal_probabilities().clone();
    let bitboards = env.get_bitboards();
    
    GameState {
        board,
        current_player,
        move_counter: env.get_move_counter(),
        total_step_counter: env.get_total_steps(),
        dead_red,
        dead_black,
        action_masks,
        reveal_probabilities,
        bitboards,
    }
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // 初始化游戏环境和状态
            let env = DarkChessEnv::new();
            app.manage(AppState {
                game: Mutex::new(env),
                opponent_type: Mutex::new(OpponentType::PvP),
            });
            Ok(())
        })
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            reset_game,
            step_game,
            bot_move,
            get_game_state,
            get_opponent_type,
            get_move_action
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}