//! MCTS (Monte Carlo Tree Search) 示例
//! 
//! 这个示例展示了如何使用游戏环境进行简单的蒙特卡洛树搜索

use banqi_3x4::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() {
    let mut env = DarkChessEnv::new();
    let mut rng = thread_rng();

    println!("=== MCTS 示例 - 简单的随机搜索策略 ===\n");

    env.reset();
    let mut step_count = 0;

    loop {
        env.print_board();

        // 获取有效动作
        let masks = env.action_masks();
        let valid_actions: Vec<usize> = masks
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| if val == 1 { Some(idx) } else { None })
            .collect();

        if valid_actions.is_empty() {
            println!("无有效动作，游戏结束");
            break;
        }

        // 简单策略：优先翻棋，否则随机移动
        let action = if let Some(&reveal_action) = valid_actions.iter().find(|&&a| a < REVEAL_ACTIONS_COUNT) {
            println!("策略: 优先翻棋");
            reveal_action
        } else {
            println!("策略: 随机移动");
            *valid_actions.choose(&mut rng).unwrap()
        };

        println!("选择动作: {}\n", action);

        match env.step(action) {
            Ok((_, _, terminated, truncated, winner)) => {
                step_count += 1;

                if terminated || truncated {
                    env.print_board();
                    println!("\n游戏结束 (步数: {})", step_count);
                    if let Some(w) = winner {
                        match w {
                            0 => println!("结果: 和棋"),
                            1 => println!("结果: 红方获胜"),
                            -1 => println!("结果: 黑方获胜"),
                            _ => println!("未知结果"),
                        }
                    }
                    break;
                }
            },
            Err(e) => panic!("错误: {}", e),
        }

        if step_count >= 100 {
            println!("达到步数限制");
            break;
        }
    }
}
