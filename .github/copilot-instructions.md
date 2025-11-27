<!-- Copilot / AI Agent Instructions for banqi_3x4 -->

This file summarizes the essential knowledge an AI coding agent should know to be productive in the banqi_3x4 repository.

Key entry points & binaries
- `src/game_env.rs` — core game logic, state representation, action masks, and debug scenarios (`setup_two_advisors`, `setup_hidden_threats`).
- `src/nn_model.rs` — PyTorch/Tch model definition; keep consistent channel/shape constants aligned to `game_env`.
- `src/mcts.rs` — Monte Carlo Tree Search implementation (including Chance Node logic). *DO NOT* change chance-node semantics without full understanding — there is a guard comment at top of the file.
- `src/ai/` — policy wrappers and the `MctsDlPolicy` (tree reuse, ModelWrapper usage) used by GUI and headless AI.
- Binaries
  - `cargo run --bin banqi-train` — training flow (data collection, DB persistence, training loop).
  - `cargo run --bin banqi-parallel-train` — **parallel training** with multi-threaded self-play and batched inference (see `PARALLEL_TRAINING.md`).
  - `cargo run --bin banqi-verify` — verify logic/PUCT path with an untrained network.
  - `cargo run --bin banqi-verify-trained -- <model.ot>` — verify a trained model (defaults to `banqi_model_latest.ot`).
  - `cargo run --bin banqi-tauri` — GUI (Tauri) that uses `src/tauri_main.rs`.
  - `cargo run --bin banqi_3x4` — a minimal demo using random policy in `src/main.rs`.

Architecture & dataflow (short)
- The environment (`DarkChessEnv`) is the source of truth. It provides `get_state()` (Observation) and `action_masks()` used by NN/MCTS.
- Observation shapes (important):
  - Board: stack frames × channels × H × W => `(STATE_STACK_SIZE=2, 8, 3, 4)` -> flattened/reshaped into `tch` as `[1, 16, 3, 4]`.
  - Scalars: `(STATE_STACK_SIZE * 56) = 112` features -> `[1, 112]`.
  - Action space: `ACTION_SPACE_SIZE = 46` with reveal actions `REVEAL_ACTIONS_COUNT = 12`.
- NN returns (policy_logits, value). Always apply the action mask before computing softmax for a valid probability distribution.
- MCTS integrates the evaluator using the `Evaluator` trait. `MctsDlPolicy` wraps model loading, evaluator and keeps a persistent MCTS instance (call `advance` after each action to reuse tree).

Critical implementation patterns and invariants
- Always use `env.action_masks()` when converting logits to probabilities. The code uses a mask technique like `masked_logits = logits + (mask - 1.0) * 1e9` to force invalid actions to -inf in practice.
- `DarkChessEnv` uses a bag model for hidden pieces (`hidden_pieces`) and Chance Nodes are created on reveal actions. `mcts.rs` handles Chance Nodes specially — do not remove the full expansion behavior or parent-child player value flipping.
- The `MCTS::step_next` function implements search tree reuse — it expects accurate environment state progress. If you change this logic, ensure `MctsDlPolicy::choose_action` consistency check still resets the tree when `env.get_total_steps()` mismatches.
- Model I/O: Saved by `VarStore::save("banqi_model_*.ot")` and loaded by `ModelWrapper::load_from_file`. The GUI expects `*.ot` files in working dir (`list_models` lists them).

Training & data persistence
- Training samples are persisted to SQLite `training_samples.db` (schema in `src/train.rs`). Board/scalars/probs/mask are serialized into byte blobs as little-endian f32 arrays.
- Sample shape expectations: Board serialized as board.as_slice() (f32) and scalars as scalars.as_slice(). If modifying `Observation`, keep serialization & deserialization consistent:
  - `train.rs` expects `Array4<f32>` shape `(2, 8, 3, 4)` flattened before storing.
- When adding new features to the input, update `nn_model.rs` and all serialization code paths in `train.rs`, `verify_*.rs`, and `ai/mcts_dl.rs`.

Build, runtime and dependency notes
- Tch crate requires a LibTorch library (or `libtorch` prebuilt). `build.rs` tries to add `-Wl,-rpath` and `-ltorch` if `DEP_TCH_LIBTORCH_LIB` is present. If you need GPU support, install a libtorch that matches your CUDA version or set `LIBTORCH`/`DEP_TCH_LIBTORCH_LIB` as required.
- Typical commands:
  - `cargo build --release` — build all binaries.
  - `cargo run --bin banqi-train` — starts training (auto-detects CUDA via `tch::Cuda::is_available()`)
  - `cargo run --bin banqi-verify-trained -- banqi_model_latest.ot` — verify trained model outputs for scenario tests
  - `cargo run --bin banqi-tauri` — starts GUI (no frontend build step; `frontend/` contains static files).
- The code uses prints for logs; for in-depth debugging look at scenarios in `game_env.rs` (`setup_two_advisors`, `setup_hidden_threats`) and `verify.rs` outputs.

Extension & contribution tips
- Adding a new Policy: implement the `Policy` trait in `src/ai/` and export it from `src/ai/mod.rs`. If your policy uses a neural net evaluator, implement `Evaluator` and hook into `MctsDlPolicy` or `MCTS` directly.
- Changing network dimensions: update `nn_model.rs`, and update all code that constructs and reshapes tensors: training (`train.rs`), inference (`verify_trained.rs`) and evaluator wrappers (`ai/mcts_dl.rs`).
- If modifying `MCTS` selection or chance-handling logic, add tests covering:
  - `Chance Node` full expansion semantics (must iterate through `hidden_pieces` possibilities)
  - `value_from_child_perspective` correctness when flipping players
  - `step_next` tree reuse in presence of reveal actions

Small code snippets the agent will often need
- Get Observation: `let obs = env.get_state();` (returns `Observation` with `board` and `scalars`).
- Action Masks: `let masks = env.action_masks();` — always use this to mask logits.
- Map action -> coords: `env.get_coords_for_action(action)`.
- Save/Load model: `vs.save("banqi_model_latest.ot")`, `vs.load(path)` via `nn::VarStore`.

Where to look next (files to inspect when changing a subsystem)
- Game rules & state — `src/game_env.rs`
- MCTS logic & chance node handling — `src/mcts.rs` (read top-of-file comment carefully)
- Network & shapes — `src/nn_model.rs`
- Training & DB I/O — `src/train.rs`
- Tauri GUI integration + model load usage — `src/tauri_main.rs` and `frontend/` files

If something is not documented here, ask for the exact workflow (e.g., target CUDA version, local LibTorch install location) and confirm before changing build scripts or CI settings.

--- End of copilot-instructions.md
