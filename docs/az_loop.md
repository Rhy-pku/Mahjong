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
