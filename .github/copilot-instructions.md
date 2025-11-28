<!-- Copilot / AI Agent Instructions for banqi_3x4 -->

This file summarizes the essential knowledge an AI coding agent should know to be productive in the banqi_3x4 repository.

## Key entry points & binaries

Core modules:
- `src/game_env.rs` — game logic, state representation (`Observation` struct), action masks, and debug scenarios (`setup_two_advisors`, `setup_hidden_threats`). **Source of truth** for all game state.
- `src/nn_model.rs` — PyTorch/Tch model (`BanqiNet`) with residual CNN architecture. Keep channel/shape constants synchronized with `game_env.rs`.
- `src/mcts.rs` — Monte Carlo Tree Search with Chance Node support for hidden pieces. **CRITICAL**: Has guard comment at top — do NOT modify chance-node full expansion or value flipping logic without deep understanding.
- `src/ai/` — policy abstractions (`Policy` trait), including `MctsDlPolicy` (persistent MCTS with tree reuse), `RandomPolicy`, `RevealFirstPolicy`.
- `src/inference.rs` — **批量推理服务器**: `InferenceServer` 收集多个推理请求并批量处理以提高GPU利用率，`ChannelEvaluator` 实现 `Evaluator` trait。
- `src/self_play.rs` — **自对弈工作器**: `SelfPlayWorker` 执行自对弈游戏，`ScenarioType` 定义特定场景，`GameEpisode` 存储游戏数据。
- `src/training.rs` — **训练步骤逻辑**: `train_step()` 执行单个训练epoch，处理批量数据和损失计算。
- `src/database.rs` — **数据库操作**: SQLite样本存储与加载（`init_database`, `save_samples_to_db`, `load_samples_from_db`）。
- `src/training_log.rs` — **训练日志**: `TrainingLog` 结构用于CSV格式的训练指标记录。
- `src/scenario_validation.rs` — **场景验证**: `validate_model_on_scenarios_with_net()` 在标准场景上评估模型表现。

Binaries (see `Cargo.toml [[bin]]` sections):
- `cargo run --bin banqi-train` — sequential training (single-threaded self-play + DB persistence).
- `cargo run --bin banqi-parallel-train` — **production training system**: multi-threaded self-play with batched GPU inference via `InferenceServer`. Uses channel-based architecture for request batching.
- `cargo run --bin banqi-verify` — verify MCTS logic with untrained network (sanity check).
- `cargo run --bin banqi-verify-trained -- <model.ot>` — verify trained model outputs on test scenarios (defaults to `banqi_model_latest.ot`).
- `cargo run --bin banqi-verify-samples` — inspect training samples from database for quality assurance.
- `cargo run --bin banqi-verify2` — alternative verification tool.
- `cargo run --bin banqi-tauri` — GUI app (default run target). Frontend is static HTML/JS in `frontend/`.
- `cargo run --bin banqi_3x4` — minimal demo with random policy.
- `cargo run --bin test-observation` — unit test for observation encoding.

## Architecture & dataflow

**Modular architecture** (重构后):
- **Game Environment** (`game_env.rs`): 环境状态管理，是所有游戏逻辑的source of truth
- **Neural Network** (`nn_model.rs`): 残差CNN架构，输入观察值，输出策略logits和价值
- **MCTS** (`mcts.rs`): 带Chance Node的蒙特卡洛树搜索
- **Inference Server** (`inference.rs`): 批量推理服务，通过channel接收请求并批量处理
- **Self-Play Workers** (`self_play.rs`): 多线程自对弈，生成训练样本
- **Training** (`training.rs`): 训练步骤逻辑，损失计算和优化
- **Database** (`database.rs`): 训练样本持久化
- **Validation** (`scenario_validation.rs`): 模型在标准场景上的评估

Environment is source of truth:
- `DarkChessEnv` provides `get_state()` → `Observation { board, scalars }` and `action_masks()`.
- Hidden pieces tracked in `hidden_pieces: Vec<Piece>` — reveal actions create Chance Nodes in MCTS.
- Player alternates via `step(action, piece_override)` — `piece_override` used only for Chance Node expansion.

Observation shapes (**critical for NN I/O**):
- Board: `Array4<f32>` shape `(STATE_STACK_SIZE, 8, 3, 4)` where `STATE_STACK_SIZE=1` (no frame stacking).
  - Flattened to `[1, 16, 3, 4]` for `tch::Tensor` input (16 = 1×8 × 2, but model uses `[1, 8, 3, 4]`).
- Scalars: `Array1<f32>` shape `(STATE_STACK_SIZE * 56,)` = 56 features → `[1, 56]` tensor.
- Action space: `ACTION_SPACE_SIZE = 46` (12 reveal + 34 move actions).

Neural network flow:
- Input: `BanqiNet::forward(board: &Tensor, scalars: &Tensor)` → `(policy_logits, value)`.
- **Always mask logits** before softmax: `masked_logits = logits + (mask - 1.0) * 1e9` (forces invalid actions to -∞).
- Value head uses `tanh` → range [-1, 1].

MCTS integration:
- `Evaluator` trait provides `(policy_probs, value)` for a state.
- `MctsDlPolicy` wraps a persistent MCTS instance — call `advance(env, action)` after each move to reuse tree.
- Tree reuse consistency: `MctsDlPolicy::choose_action` checks `env.get_total_steps()` and resets tree on mismatch.

## Critical implementation patterns and invariants

Action masking (non-negotiable):
- Always use `env.action_masks()` before sampling/selecting actions.
- Masking pattern: `masked_logits = logits + (mask.cast() - 1.0) * 1e9` (see `nn_model.rs`, `train.rs`).

MCTS Chance Node handling (**READ THE GUARD COMMENT IN `mcts.rs`**):
- Reveal actions spawn Chance Nodes that **fully expand** all possible hidden pieces.
- Each Chance Node child has probability `1.0 / hidden_pieces.len()`.
- **Player alternation**: Value sign flips when parent/child players differ — handled in `value_from_child_perspective`.
- `MCTS::step_next` reuses tree by advancing root — expects accurate environment sync.

Model I/O conventions:
- Save: `vs.save("banqi_model_<iteration>.ot")` via `nn::VarStore`.
- Load: `ModelWrapper::load_from_file(path)` in `ai/mcts_dl.rs`.
- GUI lists `*.ot` files in working directory via `list_models` command.

State stack handling:
- `STATE_STACK_SIZE=1` (disabled frame stacking per `game_env.rs` line 14).
- NN expects single frame but architecture supports multi-frame — keep constants consistent across modules.

## Training & data persistence

Database schema (`training_samples.db`):
- Table: `training_samples` with columns: `id`, `iteration`, `episode_type`, `board_state`, `scalar_state`, `policy_probs`, `value_target`, `action_mask`, `game_length`, `step_in_game`, `timestamp`.
- `board_state`: `Array4<f32>` shape `(1, 8, 3, 4)` serialized as little-endian f32 bytes.
- `scalars_state`: `Array1<f32>` shape `(56,)` serialized as little-endian f32 bytes.
- `game_length` and `step_in_game` added in parallel training — used for analyzing sample quality (e.g., selecting endgame samples).

Serialization invariants:
- Board: `obs.board.as_slice()` flattened → byte blob.
- Scalars: `obs.scalars.as_slice()` flattened → byte blob.
- **If changing `Observation` structure**, update all serialization paths: `train.rs`, `parallel_train.rs`, `verify_samples.rs`, `ai/mcts_dl.rs`.

Training flows:
- **Sequential** (`banqi-train`): Single-threaded self-play → DB insert → batch training with Adam optimizer.
- **Parallel** (`banqi-parallel-train`): Multi-worker self-play → `InferenceServer` batches GPU inference → DB insert → training epochs.
  - `InferenceServer` collects requests via `mpsc::channel`, batches up to `batch_size`, and runs forward pass on GPU.
  - Workers use `ChannelEvaluator` (implements `Evaluator`) to send inference requests and block for responses.
  - Currently focused on scenario-based training (TwoAdvisors, HiddenThreats).

**Refactored structure** (2025-11-29):
- `parallel_train.rs` reduced from 1632 lines to ~246 lines (main control loop only)
- All helper modules extracted: `inference.rs`, `self_play.rs`, `training.rs`, `database.rs`, `training_log.rs`, `scenario_validation.rs`
- Better separation of concerns and code reusability

## Build, runtime and dependency notes

LibTorch dependency:
- Tch crate requires LibTorch library. `build.rs` adds `-Wl,-rpath` and `-ltorch` linker flags if `DEP_TCH_LIBTORCH_LIB` is set.
- For GPU support, install LibTorch matching your CUDA version or set `LIBTORCH`/`DEP_TCH_LIBTORCH_LIB` environment variables.
- Training auto-detects CUDA via `tch::Cuda::is_available()` and falls back to CPU.

Common commands:
- `cargo build --release` — build all binaries with optimizations.
- `cargo run --bin banqi-train` — start sequential training.
- `cargo run --bin banqi-parallel-train` — start parallel training (recommended for production).
- `cargo run --bin banqi-verify-trained -- banqi_model_5.ot` — verify specific model checkpoint.
- `cargo run --bin banqi-tauri` — launch GUI (no frontend build step; `frontend/` contains static HTML/JS/CSS).

Debugging:
- Set `BANQI_VERBOSE=1` environment variable to enable detailed inference server logs.
- Use test scenarios in `game_env.rs`: `setup_two_advisors` (endgame), `setup_hidden_threats` (tactical).
- `verify_samples.rs` queries DB for specific sample types (e.g., samples 2 steps from game end with value=1).

## Extension & contribution tips

Adding a new Policy:
- Implement the `Policy` trait in `src/ai/` and export from `src/ai/mod.rs`.
- If using neural net evaluator, implement `Evaluator` and hook into `MctsDlPolicy` or `MCTS` directly.
- Example policies: `RandomPolicy`, `RevealFirstPolicy`, `MctsDlPolicy`.

Adding a new training scenario:
- Add variant to `ScenarioType` enum in `self_play.rs`
- Implement setup function in `game_env.rs` (e.g., `setup_xxx()`)
- Update `ScenarioType::create_env()` and related methods

Changing network dimensions:
- Update constants in `nn_model.rs`: `BOARD_CHANNELS`, `STATE_STACK`, `SCALAR_FEATURES`, `ACTION_SIZE`.
- Update all tensor construction/reshape code: `training.rs`, `inference.rs`, `scenario_validation.rs`, `ai/mcts_dl.rs`.
- Verify consistency with `game_env.rs` constants: `STATE_STACK_SIZE`, `ACTION_SPACE_SIZE`.

Modifying MCTS logic:
- **DO NOT** change Chance Node expansion semantics without reading guard comment at top of `mcts.rs`.
- Add tests covering:
  - Chance Node full expansion (must iterate all `hidden_pieces` possibilities).
  - `value_from_child_perspective` correctness when flipping players.
  - `step_next` tree reuse with reveal actions.

Parallel training tuning:
- Adjust `batch_size` and `batch_timeout_ms` in `InferenceServer` for GPU utilization vs latency tradeoff.
- Worker count defaults to number of scenarios (currently 2: TwoAdvisors + HiddenThreats).
- Monitor console output for loss trends and scenario verification metrics.

## Small code snippets the agent will often need

Get Observation:
```rust
let obs = env.get_state();  // returns Observation with board and scalars
```

Action Masks:
```rust
let masks = env.action_masks();  // always use this to mask logits
```

Map action to coordinates:
```rust
let coords = env.get_coords_for_action(action);
```

Save/Load model:
```rust
vs.save("banqi_model_latest.ot")?;
vs.load(path)?;  // via nn::VarStore
```

Mask logits before softmax:
```rust
let masked_logits = logits + (mask.cast() - 1.0) * 1e9;
let probs = masked_logits.softmax(-1, Kind::Float);
```


## Where to look next (files to inspect when changing a subsystem)

- Game rules & state — `src/game_env.rs`
- MCTS logic & chance node handling — `src/mcts.rs` (read top-of-file comment carefully)
- Network & shapes — `src/nn_model.rs`
- Training steps & loss calculation — `src/training.rs`
- Inference server & batching — `src/inference.rs`
- Self-play workers & scenarios — `src/self_play.rs`
- Database persistence — `src/database.rs`
- Scenario validation — `src/scenario_validation.rs`
- Parallel training main loop — `src/parallel_train.rs`
- Sequential training — `src/train.rs`
- Tauri GUI integration + model load usage — `src/tauri_main.rs` and `frontend/` files

If something is not documented here, ask for the exact workflow (e.g., target CUDA version, local LibTorch install location) and confirm before changing build scripts or CI settings.

--- End of copilot-instructions.md
