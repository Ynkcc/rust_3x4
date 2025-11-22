// Tauri 应用入口文件
use banqi_3x4::*;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{Manager, State};

// 游戏状态的可序列化版本
#[derive(Debug, Clone, Serialize)]
struct GameState {
    board: Vec<String>,  // 每个格子的状态: "Empty", "Hidden", "R_Sol" 等
    current_player: String,
    move_counter: usize,
    total_step_counter: usize,
    dead_red: Vec<String>,
    dead_black: Vec<String>,
    action_masks: Vec<i32>,
    reveal_probabilities: Vec<f32>,
    bitboards: HashMap<String, Vec<bool>>,  // Bitboards 可视化数据
}

#[derive(Debug, Clone, Serialize)]
struct StepResult {
    state: GameState,
    terminated: bool,
    truncated: bool,
    winner: Option<i32>,
}

// 全局游戏环境（使用 Mutex 保证线程安全）
struct GameEnv(Mutex<DarkChessEnv>);

// Tauri 命令：重置游戏
#[tauri::command]
fn reset_game(env: State<GameEnv>) -> GameState {
    let mut game = env.inner().0.lock().unwrap();
    game.reset();
    extract_game_state(&*game)
}

// Tauri 命令：执行动作
#[tauri::command]
fn step_game(action: usize, env: State<GameEnv>) -> Result<StepResult, String> {
    let mut game = env.inner().0.lock().unwrap();
    
    match game.step(action) {
        Ok((_obs, _reward, terminated, truncated, winner)) => {
            let state = extract_game_state(&*game);
            Ok(StepResult {
                state,
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
fn get_game_state(env: State<GameEnv>) -> GameState {
    let game = env.inner().0.lock().unwrap();
    extract_game_state(&*game)
}

// Tauri 命令：获取移动动作编号
#[tauri::command]
fn get_move_action(from_sq: usize, to_sq: usize, env: State<GameEnv>) -> Option<usize> {
    let game = env.inner().0.lock().unwrap();
    game.get_action_for_coords(&vec![from_sq, to_sq])
}

// 辅助函数：获取棋子短名称
fn get_piece_short_name(piece: &Piece) -> String {
    // 复制 Piece::short_name 的逻辑
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

//#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // 初始化游戏环境
            let env = DarkChessEnv::new();
            app.manage(GameEnv(Mutex::new(env)));
            Ok(())
        })
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            reset_game,
            step_game,
            get_game_state,
            get_move_action
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}
