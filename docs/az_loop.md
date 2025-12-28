AlphaZero Loop (az_loop)

Directory Layout
- runs/az/{run_name}/
- runs/az/{run_name}/state.json
- runs/az/{run_name}/models/best.pt
- runs/az/{run_name}/models/candidates/iter_0001.pt
- runs/az/{run_name}/data/iter_0001/...
- runs/az/{run_name}/data/mix_iter_0001/...
- runs/az/{run_name}/arena/iter_0001.json
- runs/az/{run_name}/logs/...

Quick Start
- Basic run:
  - python az_loop.py --run_name exp01 --init_model teacher.pt
- Resume:
  - python az_loop.py --run_name exp01 --resume
- Smoke test:
  - python az_loop.py --run_name exp01 --init_model teacher.pt --dry_run

Config Override
- JSON or YAML:
  - python az_loop.py --run_name exp01 --config configs/az_loop.json

Recovery
- state.json records iteration, best model path, and latest arena result.
- After crash, use --resume to continue from last completed iteration.

Core Flow
- self-play -> train -> arena -> accept/rollback -> repeat
- Candidate replaces best only if arena acceptance criteria are met.

Notes
- self-play data is stored in data/iter_xxxx
- training mixes the most recent data_window iterations
- arena uses score mean + confidence interval by default
- train uses reward_scale (default 100.0); if your self-play data is already scaled, set train.reward_scale=1
- train can cap replay buffer size by samples via train.replay_buffer_samples (None disables)
- train can use cosine LR via train.lr_schedule=cosine with train.lr_min and train.warmup_steps
- train DDP uses train.ddp_master_port to avoid port conflicts when multiple runs are active
- self-play prints periodic progress (npz count and size) via self_play.progress_interval_sec
- per-worker progress is aggregated via self_play.worker_progress_interval_sec
- self-play supports pruning via top_k/min_actions/policy_mass to speed up MCTS
- self-play supports batched leaf eval via self_play.leaf_batch_size
- self-play can round-robin GPUs via self_play.gpu_ids (e.g., [0,1,2,3])
- self-play can choose MCTS mode via self_play.mcts_mode (all|single|subset); in subset mode, use self_play.mcts_players and self_play.mcts_player_rotate
- self-play can delay MCTS until late game via self_play.start_wall_limit (remaining wall tiles)
- self-play can enable forced playouts via self_play.forced_playout_k and prune policy targets via self_play.policy_prune_min_visits
- self-play can add root Dirichlet noise via self_play.root_dirichlet_alpha and self_play.root_exploration_fraction
- train can use DDP via train.ddp=true and train.gpu_ids
