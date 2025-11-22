const { invoke } = window.__TAURI__.core;

let selectedSquare = null; // 记录当前选中的格子索引 (0-15)
let gameState = null;

// 棋子文字映射
function getPieceText(slot) {
  if (slot.type === "Empty") return "";
  if (slot.type === "Hidden") return "?";
  if (slot.type === "Revealed") {
    // Rust 返回的 data 是 Piece 结构 { piece_type: "General", player: "Red" }
    const p = slot.data;
    const isRed = p.player === "Red";
    const type = p.piece_type;
    if (isRed) {
      if (type === "General") return "帥";
      if (type === "Advisor") return "仕";
      if (type === "Soldier") return "兵";
    } else {
      if (type === "General") return "將";
      if (type === "Advisor") return "士";
      if (type === "Soldier") return "卒";
    }
  }
  return "";
}

// 初始化棋盘 DOM
function initBoard() {
  const boardEl = document.getElementById('chess-board');
  boardEl.innerHTML = '';
  for (let i = 0; i < 12; i++) { // 12个格子 (3x4)? 不, 代码里是 3行4列 = 12个位置？
    // 注意：Rust 中 TOTAL_POSITIONS = 12。
    // 如果你的 Rust 代码 BOARD_ROWS=3, BOARD_COLS=4，那确实是 12。
    // 但 PySide 代码里看起来是 4x4=16。
    // 请检查 Rust 中的 BOARD_ROWS。这里我假设用 Rust 的 12。
    // *修正*：看 PySide 是 4x4。看 Rust 是 3x4。
    // 我将遵循 Rust main.rs 中的常量: const BOARD_ROWS: usize = 3;
    // 为了对齐 UI，我建议将 Rust 中的 BOARD_ROWS 改为 4 适配标准暗棋，
    // 或者这里前端动态生成。
  }
  // 动态生成
}

async function updateUI(state) {
  gameState = state;
  
  // 1. 更新日志
  const logArea = document.getElementById('log-area');
  logArea.value = state.logs.join('\n');
  logArea.scrollTop = logArea.scrollHeight;

  // 2. 更新状态栏
  document.getElementById('current-player').value = state.current_player;
  document.getElementById('game-status').value = state.winner;
  document.getElementById('move-counter').value = state.move_counter;
  document.getElementById('dead-red').textContent = state.dead_red.join(', ') || "无";
  document.getElementById('dead-black').textContent = state.dead_black.join(', ') || "无";

  // 3. 渲染棋盘
  const boardEl = document.getElementById('chess-board');
  boardEl.innerHTML = ''; // 清空重绘
  // Rust是3x4，需要调整CSS grid-template-rows
  boardEl.style.gridTemplateRows = `repeat(${state.board.length / 4}, 1fr)`; // 假设宽是4

  state.board.forEach((slot, idx) => {
    const cell = document.createElement('div');
    cell.className = 'chess-cell';
    
    // 样式类
    if (slot.type === "Hidden") cell.classList.add('hidden');
    if (slot.type === "Empty") cell.classList.add('empty');
    if (slot.type === "Revealed") {
      cell.classList.add(slot.data.player === "Red" ? 'red' : 'black');
    }
    if (selectedSquare === idx) cell.classList.add('selected');

    cell.textContent = getPieceText(slot);
    cell.onclick = () => onSquareClick(idx);
    boardEl.appendChild(cell);
  });

  // 4. 渲染 Bitboards
  renderBitboards(state.bitboards);
}

function renderBitboards(bbs) {
  const container = document.getElementById('bitboard-container');
  container.innerHTML = '';
  
  // 固定的顺序
  const keys = ["hidden", "empty", "R_rev", "B_rev", "R_sol", "B_sol"]; 
  
  for (const key of Object.keys(bbs)) {
    const wrapper = document.createElement('div');
    wrapper.className = 'bb-wrapper';
    
    const label = document.createElement('div');
    label.className = 'bb-label';
    label.textContent = key;
    
    const grid = document.createElement('div');
    grid.className = 'bb-grid';
    
    bbs[key].forEach(isActive => {
      const dot = document.createElement('div');
      dot.className = `bb-cell ${isActive ? 'active' : ''}`;
      grid.appendChild(dot);
    });
    
    wrapper.appendChild(label);
    wrapper.appendChild(grid);
    container.appendChild(wrapper);
  }
}

async function onSquareClick(idx) {
  if (gameState.game_over) return;

  const slot = gameState.board[idx];
  const myPlayerVal = gameState.player_val; // 1 or -1

  // 逻辑：
  // 1. 如果当前未选中:
  //    - 点击 Hidden -> 发送 action (翻牌)
  //    - 点击 自己的 Revealed -> 选中 (Select)
  // 2. 如果已选中 (Selected):
  //    - 点击 目标 -> 发送 action (尝试移动 selected -> idx)
  //    - 点击 同样位置 -> 取消选中
  //    - 点击 自己其他棋子 -> 切换选中

  if (selectedSquare === null) {
    if (slot.type === "Hidden") {
      // 尝试翻开
      try {
        const newState = await invoke("handle_interaction", { fromSq: null, toSq: idx });
        updateUI(newState);
      } catch (e) {
        console.error("Action failed:", e); // 可能是非当前玩家回合
      }
    } else if (slot.type === "Revealed") {
      // 检查是否是当前玩家的棋子
      // Rust Player: Red=1, Black=-1
      const pVal = slot.data.player === "Red" ? 1 : -1;
      if (pVal === myPlayerVal) {
        selectedSquare = idx;
        updateUI(gameState); // 重绘以显示高亮
      }
    }
  } else {
    // 已有选中
    if (idx === selectedSquare) {
      selectedSquare = null; // 取消
      updateUI(gameState);
    } else if (slot.type === "Revealed" && (slot.data.player === (myPlayerVal === 1 ? "Red" : "Black"))) {
      selectedSquare = idx; // 切换选中
      updateUI(gameState);
    } else {
      // 尝试移动 (攻击或移动到空位)
      try {
        const newState = await invoke("handle_interaction", { fromSq: selectedSquare, toSq: idx });
        selectedSquare = null;
        updateUI(newState);
      } catch (e) {
        console.log("Invalid move:", e);
        // 移动非法，不取消选中，让用户重试
      }
    }
  }
}

// 启动
window.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('btn-new-game').onclick = async () => {
    selectedSquare = null;
    const state = await invoke("new_game");
    updateUI(state);
  };

  // 加载初始状态
  const state = await invoke("new_game");
  updateUI(state);
});