#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaZero training loop for oracle teacher (self-play -> train -> arena -> accept/rollback).
python az_loop.py --run_name exp01 --config configs/az_loop.json --init_model teacher_initial.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from model_pretrain import PretrainModel

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

from mcts_self_play import build_oracle_obs, load_state_dict_compat


RUNS_ROOT = os.path.join("runs", "az")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    print("[%s] %s" % (_now(), msg), flush=True)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError("config not found: %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if path.endswith(".json"):
        return json.loads(raw)
    if path.endswith(".yml") or path.endswith(".yaml"):
        if yaml is None:
            raise RuntimeError("PyYAML not installed, cannot read yaml config")
        return yaml.safe_load(raw)
    raise ValueError("unsupported config file: %s" % path)


def safe_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_config() -> Dict[str, Any]:
    cpu = os.cpu_count() or 8
    return {
        "seed": 2025,
        "max_iters": 1000000,
        "data_window": 5,
        "cleanup": {
            "keep_data_iters": 5,
            "keep_models": 5,
        },
        "model": {
            "hidden_dim": 256,
            "in_channels": 106,
            "value_dim": 4,
        },
        "self_play": {
            "episodes": 200,
            "workers": max(1, cpu - 1),
            "device": "cuda",
            "gpu_ids": [],
            "simulations": 64,
            "c_puct": 1.5,
            "reward_scale": 1.0,
            "determinize": True,
            "temperature": 1.0,
            "temperature_drop_iter": 10,
            "save_every": 4096,
            "log_interval": 1,
            "progress_interval_sec": 60,
            "worker_progress_interval_sec": 30,
            "top_k": 12,
            "min_actions": 6,
            "policy_mass": 0.95,
            "leaf_batch_size": 16,
            "mcts_mode": "all",
            "mcts_player": -1,
            "mcts_player_rotate": False,
            "mcts_players": [],
            "start_wall_limit": 0,
            "timeout_sec": 3600 * 6,
            "retries": 1,
            "wandb": False,
            "wandb_project": "mahjong-az",
            "wandb_run_name": "",
            "wandb_log_interval": 10,
        },
        "train": {
            "device": "cuda",
            "epochs": 2,
            "batch_size": 1024,
            "lr": 1e-4,
            "lr_schedule": "constant",
            "lr_min": 0.0,
            "warmup_steps": 0,
            "value_weight": 1.0,
            "policy_weight": 1.0,
            "reward_scale": 100.0,
            "num_workers": max(1, cpu // 2),
            "prefetch_factor": 2,
            "steps_per_epoch": None,
            "replay_buffer_samples": None,
            "seed": None,
            "amp": True,
            "dataloader_timeout": 0,
            "timeout_sec": 3600 * 6,
            "retries": 1,
            "ddp": False,
            "ddp_backend": "nccl",
            "gpu_ids": [],
            "wandb": False,
            "wandb_project": "mahjong-az",
            "wandb_run_name": "",
            "wandb_log_interval": 50,
            "progress_interval": 50,
        },
        "arena": {
            "episodes": 200,
            "min_episodes": 50,
            "check_interval": 20,
            "workers": max(1, cpu // 2),
            "device": "cpu",
            "reward_scale": 1.0,
            "seed": 2026,
            "timeout_sec": 3600,
            "retries": 0,
        },
        "accept": {
            "metric": "score",
            "score_threshold": 0.0,
            "winrate_threshold": 0.55,
            "winrate_lower_bound": 0.50,
            "confidence": 0.95,
        },
    }


def prepare_run_dir(run_name: str) -> Dict[str, str]:
    run_dir = os.path.join(RUNS_ROOT, run_name)
    paths = {
        "run": run_dir,
        "models": os.path.join(run_dir, "models"),
        "best": os.path.join(run_dir, "models", "best.pt"),
        "candidates": os.path.join(run_dir, "models", "candidates"),
        "data": os.path.join(run_dir, "data"),
        "arena": os.path.join(run_dir, "arena"),
        "logs": os.path.join(run_dir, "logs"),
        "state": os.path.join(run_dir, "state.json"),
    }
    for key in ("models", "candidates", "data", "arena", "logs"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def init_best_model(paths: Dict[str, str], cfg: Dict[str, Any], init_model: Optional[str]) -> str:
    if init_model:
        shutil.copy2(init_model, paths["best"])
        return paths["best"]
    if os.path.exists(paths["best"]):
        return paths["best"]
    model_cfg = cfg["model"]
    model = PretrainModel(
        hidden_dim=model_cfg["hidden_dim"],
        use_vec=False,
        in_channels=model_cfg["in_channels"],
        value_dim=model_cfg["value_dim"],
    )
    torch.save(model.state_dict(), paths["best"])
    return paths["best"]


def load_or_init_state(paths: Dict[str, str], cfg: Dict[str, Any], init_model: Optional[str]) -> Dict[str, Any]:
    if os.path.exists(paths["state"]):
        return read_json(paths["state"])
    best_path = init_best_model(paths, cfg, init_model)
    state = {
        "iter": 0,
        "best_model": best_path,
        "last_arena": None,
        "last_update": _now(),
    }
    safe_write_json(paths["state"], state)
    return state


def run_command(
    cmd: List[str],
    log_path: str,
    timeout_sec: int,
    retries: int,
    stage: str,
    env_override: Optional[Dict[str, str]] = None,
) -> None:
    for attempt in range(retries + 1):
        log("run %s (attempt %d/%d): %s" % (stage, attempt + 1, retries + 1, " ".join(cmd)))
        start = time.time()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("[%s] CMD: %s\n" % (_now(), " ".join(cmd)))
            f.flush()
            env = os.environ.copy()
            if env_override:
                env.update(env_override)
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
                print(line, end="")
                if timeout_sec > 0 and time.time() - start > timeout_sec:
                    proc.kill()
                    raise RuntimeError("%s timeout after %d sec" % (stage, timeout_sec))
            ret = proc.wait()
        if ret == 0:
            return
        if attempt == retries:
            raise RuntimeError("%s failed with code %d" % (stage, ret))
        time.sleep(3)


def list_npz_files(dir_path: str) -> List[str]:
    out = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.endswith(".npz"):
                out.append(os.path.join(root, name))
    return sorted(out)


def count_npz_samples(dir_path: str) -> int:
    total = 0
    for path in list_npz_files(dir_path):
        with np.load(path, mmap_mode="r") as data:
            total += int(data["oracle_obs"].shape[0])
    return total


def npz_sample_count(path: str) -> int:
    with np.load(path, mmap_mode="r") as data:
        return int(data["oracle_obs"].shape[0])


def count_npz_files(dir_path: str) -> Tuple[int, float]:
    total_files = 0
    total_bytes = 0
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.endswith(".npz"):
                total_files += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return total_files, total_bytes / (1024 * 1024)


def read_worker_progress(paths: Dict[str, str], iter_id: int, workers: int) -> Tuple[int, int]:
    total = 0
    done = 0
    for worker_id in range(workers):
        progress_path = os.path.join(paths["logs"], "selfplay_iter_%04d_worker_%d.progress" % (iter_id, worker_id))
        if not os.path.exists(progress_path):
            continue
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                val = int(f.read().strip() or 0)
            total += val
            done += 1
        except Exception:
            continue
    return total, done


def read_worker_summary(paths: Dict[str, str], iter_id: int, workers: int) -> Dict[str, int]:
    totals = {
        "episodes": 0,
        "mcts_episodes": 0,
        "mcts_wins": 0,
        "non_mcts_wins": 0,
        "draws": 0,
        "invalid": 0,
    }
    for worker_id in range(workers):
        summary_path = os.path.join(paths["logs"], "selfplay_iter_%04d_worker_%d.summary.json" % (iter_id, worker_id))
        if not os.path.exists(summary_path):
            continue
        try:
            data = read_json(summary_path)
        except Exception:
            continue
        for key in totals:
            try:
                totals[key] += int(data.get(key, 0))
            except Exception:
                continue
    return totals


def ensure_clean_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def spawn_self_play(
    cfg: Dict[str, Any],
    paths: Dict[str, str],
    iter_id: int,
    best_model: str,
) -> str:
    sp_cfg = cfg["self_play"]
    iter_dir = os.path.join(paths["data"], "iter_%04d" % iter_id)
    ensure_clean_dir(iter_dir)

    episodes = int(sp_cfg["episodes"])
    workers = int(sp_cfg["workers"])
    workers = max(1, min(workers, episodes))
    per_worker = [episodes // workers] * workers
    for i in range(episodes % workers):
        per_worker[i] += 1

    def pick_worker_device(worker_id: int) -> str:
        device = str(sp_cfg.get("device", "cpu"))
        if not device.startswith("cuda"):
            return device
        if ":" in device:
            return device
        gpu_ids = sp_cfg.get("gpu_ids") or []
        if gpu_ids:
            return "cuda:%d" % gpu_ids[worker_id % len(gpu_ids)]
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return "cuda:%d" % (worker_id % count)
        return device

    def worker_cmd(worker_id: int, ep: int) -> Tuple[List[str], str]:
        out_dir = os.path.join(iter_dir, "worker_%d" % worker_id)
        os.makedirs(out_dir, exist_ok=True)
        progress_path = os.path.join(paths["logs"], "selfplay_iter_%04d_worker_%d.progress" % (iter_id, worker_id))
        summary_path = os.path.join(paths["logs"], "selfplay_iter_%04d_worker_%d.summary.json" % (iter_id, worker_id))
        worker_device = pick_worker_device(worker_id)
        cmd = [
            sys.executable,
            "mcts_self_play.py",
            "--model",
            best_model,
            "--episodes",
            str(ep),
            "--device",
            worker_device,
            "--simulations",
            str(sp_cfg["simulations"]),
            "--c_puct",
            str(sp_cfg["c_puct"]),
            "--reward_scale",
            str(sp_cfg["reward_scale"]),
            "--temperature",
            str(sp_cfg["temperature"]),
            "--top_k",
            str(sp_cfg.get("top_k", 0)),
            "--min_actions",
            str(sp_cfg.get("min_actions", 6)),
            "--policy_mass",
            str(sp_cfg.get("policy_mass", 0.95)),
            "--leaf_batch_size",
            str(sp_cfg.get("leaf_batch_size", 16)),
            "--mcts_mode",
            str(sp_cfg.get("mcts_mode", "all")),
            "--mcts_player",
            str(sp_cfg.get("mcts_player", -1)),
            "--start_wall_limit",
            str(sp_cfg.get("start_wall_limit", 0)),
            "--out_dir",
            out_dir,
            "--save_every",
            str(sp_cfg["save_every"]),
            "--log_interval",
            str(sp_cfg["log_interval"]),
            "--progress_path",
            progress_path,
            "--progress_interval_sec",
            str(sp_cfg.get("worker_progress_interval_sec", 30)),
            "--summary_path",
            summary_path,
            "--seed",
            str(cfg["seed"] + iter_id * 1000 + worker_id),
        ]
        if sp_cfg["determinize"]:
            cmd.append("--determinize")
        if sp_cfg.get("mcts_player_rotate"):
            cmd.append("--mcts_player_rotate")
        mcts_players = sp_cfg.get("mcts_players")
        if mcts_players:
            if isinstance(mcts_players, (list, tuple)):
                mcts_players_arg = ",".join(str(p) for p in mcts_players)
            else:
                mcts_players_arg = str(mcts_players)
            cmd += ["--mcts_players", mcts_players_arg]
        if sp_cfg.get("wandb"):
            cmd.append("--wandb")
            cmd += ["--wandb_project", sp_cfg["wandb_project"]]
            if sp_cfg.get("wandb_run_name"):
                cmd += ["--wandb_run_name", sp_cfg["wandb_run_name"]]
            cmd += ["--wandb_log_interval", str(sp_cfg["wandb_log_interval"])]
        log_path = os.path.join(paths["logs"], "selfplay_iter_%04d_worker_%d.log" % (iter_id, worker_id))
        return cmd, log_path

    log("self-play workers=%d episodes=%d iter=%d" % (workers, episodes, iter_id))
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for worker_id, ep in enumerate(per_worker):
            cmd, log_path = worker_cmd(worker_id, ep)
            futures.append(
                executor.submit(
                    run_command,
                    cmd,
                    log_path,
                    sp_cfg["timeout_sec"],
                    sp_cfg["retries"],
                    "self_play",
                )
            )
        pending = list(futures)
        interval = int(sp_cfg.get("progress_interval_sec", 60))
        next_report = time.time() + max(interval, 1)
        start = time.time()
        while pending:
            done = [f for f in pending if f.done()]
            for f in done:
                f.result()
                pending.remove(f)
            now = time.time()
            if interval > 0 and now >= next_report:
                npz_files, npz_mb = count_npz_files(iter_dir)
                completed, prog_workers = read_worker_progress(paths, iter_id, workers)
                log(
                    "self-play progress: done=%d/%d episodes=%d/%d npz=%d size=%.1fMB elapsed=%.1fs"
                    % (
                        workers - len(pending),
                        workers,
                        completed,
                        episodes,
                        npz_files,
                        npz_mb,
                        now - start,
                    )
                )
                next_report = now + interval
            time.sleep(1)

    total_samples = count_npz_samples(iter_dir)
    summary = read_worker_summary(paths, iter_id, workers)
    mcts_ep = summary.get("mcts_episodes", 0)
    denom = max(mcts_ep, 1)
    mcts_rate = summary.get("mcts_wins", 0) / denom
    non_mcts_rate = summary.get("non_mcts_wins", 0) / denom
    log(
        "self-play summary: episodes=%d mcts_ep=%d wins_mcts=%d wins_non_mcts=%d win=%.3f/%.3f draws=%d invalid=%d"
        % (
            summary.get("episodes", 0),
            mcts_ep,
            summary.get("mcts_wins", 0),
            summary.get("non_mcts_wins", 0),
            mcts_rate,
            non_mcts_rate,
            summary.get("draws", 0),
            summary.get("invalid", 0),
        )
    )
    log("self-play done: iter=%d total_samples=%d data_dir=%s" % (iter_id, total_samples, iter_dir))
    return iter_dir


def build_mix_dir(
    paths: Dict[str, str],
    iter_id: int,
    data_iters: List[int],
    max_samples: Optional[int] = None,
) -> str:
    mix_dir = os.path.join(paths["data"], "mix_iter_%04d" % iter_id)
    ensure_clean_dir(mix_dir)
    counter = 0
    kept_samples = 0
    if max_samples is not None and max_samples <= 0:
        max_samples = None
    iter_order = list(data_iters)
    if max_samples is not None:
        iter_order = list(reversed(iter_order))
    for it in iter_order:
        data_dir = os.path.join(paths["data"], "iter_%04d" % it)
        files = list_npz_files(data_dir)
        if max_samples is not None:
            files = list(reversed(files))
        for src in files:
            if max_samples is not None and kept_samples >= max_samples:
                break
            try:
                sample_count = npz_sample_count(src)
            except Exception:
                sample_count = 0
            dst = os.path.join(mix_dir, "iter_%04d_%08d.npz" % (it, counter))
            try:
                os.link(src, dst)
            except Exception:
                os.symlink(src, dst)
            counter += 1
            kept_samples += sample_count
        if max_samples is not None and kept_samples >= max_samples:
            break
    log("mix data: %d iters -> %d files samples=%d" % (len(data_iters), counter, kept_samples))
    return mix_dir


def train_candidate(
    cfg: Dict[str, Any],
    paths: Dict[str, str],
    iter_id: int,
    mix_dir: str,
) -> str:
    tr_cfg = cfg["train"]
    candidate_path = os.path.join(paths["candidates"], "iter_%04d.pt" % iter_id)
    cmd = [
        "train_az_oracle.py",
        "--data_dir",
        mix_dir,
        "--device",
        tr_cfg["device"],
        "--hidden_dim",
        str(cfg["model"]["hidden_dim"]),
        "--epochs",
        str(tr_cfg["epochs"]),
        "--batch_size",
        str(tr_cfg["batch_size"]),
        "--lr",
        str(tr_cfg["lr"]),
        "--lr_schedule",
        str(tr_cfg.get("lr_schedule", "constant")),
        "--lr_min",
        str(tr_cfg.get("lr_min", 0.0)),
        "--warmup_steps",
        str(tr_cfg.get("warmup_steps", 0)),
        "--value_weight",
        str(tr_cfg["value_weight"]),
        "--policy_weight",
        str(tr_cfg["policy_weight"]),
        "--reward_scale",
        str(tr_cfg["reward_scale"]),
        "--save_path",
        candidate_path,
        "--num_workers",
        str(tr_cfg["num_workers"]),
        "--prefetch_factor",
        str(tr_cfg["prefetch_factor"]),
        "--progress_interval",
        str(tr_cfg["progress_interval"]),
        "--dataloader_timeout",
        str(tr_cfg["dataloader_timeout"]),
    ]
    if tr_cfg.get("steps_per_epoch") is not None:
        cmd += ["--steps_per_epoch", str(tr_cfg["steps_per_epoch"])]
    if tr_cfg.get("seed") is not None:
        cmd += ["--seed", str(tr_cfg["seed"])]
    if tr_cfg.get("amp"):
        cmd.append("--amp")
    if tr_cfg.get("wandb"):
        cmd.append("--wandb")
        cmd += ["--wandb_project", tr_cfg["wandb_project"]]
        if tr_cfg.get("wandb_run_name"):
            cmd += ["--wandb_run_name", tr_cfg["wandb_run_name"]]
        cmd += ["--wandb_log_interval", str(tr_cfg["wandb_log_interval"])]
    env_override = None
    use_ddp = bool(tr_cfg.get("ddp")) and str(tr_cfg.get("device", "")).startswith("cuda")
    if use_ddp:
        gpu_ids = tr_cfg.get("gpu_ids") or []
        if gpu_ids:
            nproc = len(gpu_ids)
        elif torch.cuda.is_available():
            nproc = torch.cuda.device_count()
        else:
            nproc = 0
        if nproc > 1:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(nproc),
                cmd[0],
                "--ddp",
                "--ddp_backend",
                str(tr_cfg.get("ddp_backend", "nccl")),
            ] + cmd[1:]
            if gpu_ids:
                env_override = {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_ids)}
        else:
            use_ddp = False
    if not use_ddp:
        cmd = [sys.executable] + cmd
    log_path = os.path.join(paths["logs"], "train_iter_%04d.log" % iter_id)
    run_command(cmd, log_path, tr_cfg["timeout_sec"], tr_cfg["retries"], "train", env_override)
    return candidate_path


def wilson_ci(wins: int, n: int, z: float) -> Tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 1.0
    phat = wins / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return center, center - margin, center + margin


def mean_ci(values: List[float], z: float) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float32)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    std = float(arr.std(ddof=1))
    se = std / math.sqrt(len(arr))
    return mean, mean - z * se, mean + z * se


@dataclass
class ArenaResult:
    score_mean: float
    score_ci_low: float
    score_ci_high: float
    win_rate: float
    win_ci_low: float
    win_ci_high: float
    wins: int
    losses: int
    draws: int
    invalid: int
    episodes: int


def _arena_worker(
    cand_path: str,
    best_path: str,
    episodes: int,
    seed: int,
    device: str,
    reward_scale: float,
    seat_offset: int,
) -> Tuple[List[float], int, int, int, int]:
    from env import MahjongGBEnv
    from feature_10m import FeatureAgent10M as FeatureAgent

    rng = np.random.RandomState(seed)

    cand_state = load_state_dict_compat(cand_path)
    best_state = load_state_dict_compat(best_path)

    in_channels = cand_state["pre_conv.0.weight"].shape[1]
    hidden_dim = cand_state["fusion.0.weight"].shape[0]
    value_dim = cand_state["value_head.2.weight"].shape[0]

    device_t = torch.device(device)

    cand = PretrainModel(hidden_dim=hidden_dim, use_vec=False, in_channels=in_channels, value_dim=value_dim).to(device_t)
    best = PretrainModel(hidden_dim=hidden_dim, use_vec=False, in_channels=in_channels, value_dim=value_dim).to(device_t)
    cand.load_state_dict(cand_state, strict=True)
    best.load_state_dict(best_state, strict=True)
    cand.eval()
    best.eval()

    env = MahjongGBEnv(config={"agent_clz": FeatureAgent})
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    score_diffs: List[float] = []
    wins = 0
    losses = 0
    draws = 0
    invalid = 0

    for i in range(episodes):
        cand_seats = pairs[(i + seat_offset) % len(pairs)]
        obs = env.reset()
        done = False
        rewards = None
        while not done:
            obs_items = list(obs.items())
            cand_batch = []
            cand_masks = []
            cand_names = []
            best_batch = []
            best_masks = []
            best_names = []
            for name, o in obs_items:
                player = int(name.split("_")[-1]) - 1
                if player in cand_seats:
                    cand_names.append(name)
                    cand_batch.append(build_oracle_obs(env, player, o["observation"], in_channels))
                    cand_masks.append(o["action_mask"])
                else:
                    best_names.append(name)
                    best_batch.append(build_oracle_obs(env, player, o["observation"], in_channels))
                    best_masks.append(o["action_mask"])

            actions = {name: 0 for name in env.agent_names}
            if cand_batch:
                obs_t = torch.tensor(np.stack(cand_batch), dtype=torch.float32, device=device_t)
                mask_t = torch.tensor(np.stack(cand_masks), dtype=torch.float32, device=device_t)
                with torch.inference_mode():
                    logits, _ = cand(obs_t)
                masked_logits = logits + torch.clamp(torch.log(mask_t + 1e-45), min=-1e38, max=0)
                cand_act = torch.argmax(masked_logits, dim=1).cpu().numpy().tolist()
                for name, act in zip(cand_names, cand_act):
                    actions[name] = int(act)

            if best_batch:
                obs_t = torch.tensor(np.stack(best_batch), dtype=torch.float32, device=device_t)
                mask_t = torch.tensor(np.stack(best_masks), dtype=torch.float32, device=device_t)
                with torch.inference_mode():
                    logits, _ = best(obs_t)
                masked_logits = logits + torch.clamp(torch.log(mask_t + 1e-45), min=-1e38, max=0)
                best_act = torch.argmax(masked_logits, dim=1).cpu().numpy().tolist()
                for name, act in zip(best_names, best_act):
                    actions[name] = int(act)

            obs, rewards, done = env.step(actions)

        if rewards is None:
            continue
        reward_vals = [rewards.get(name, 0) for name in env.agent_names]
        if any(r == -30 for r in reward_vals):
            invalid += 1
            continue
        cand_sum = sum(reward_vals[i] for i in cand_seats) / reward_scale
        best_sum = sum(reward_vals[i] for i in range(4) if i not in cand_seats) / reward_scale
        diff = cand_sum - best_sum
        score_diffs.append(diff)
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            draws += 1

    return score_diffs, wins, losses, draws, invalid


def run_arena(
    cfg: Dict[str, Any],
    paths: Dict[str, str],
    iter_id: int,
    best_path: str,
    cand_path: str,
) -> ArenaResult:
    arena_cfg = cfg["arena"]
    episodes = int(arena_cfg["episodes"])
    min_episodes = int(arena_cfg["min_episodes"])
    check_interval = int(arena_cfg["check_interval"])
    workers = int(arena_cfg["workers"])
    reward_scale = float(arena_cfg["reward_scale"])
    seed = int(arena_cfg["seed"]) + iter_id * 100
    device = arena_cfg["device"]

    if device == "cuda" and workers > 1:
        log("arena: device=cuda with workers>1 is unstable, forcing workers=1")
        workers = 1

    z = 1.96
    score_vals: List[float] = []
    wins = 0
    losses = 0
    draws = 0
    invalid = 0

    remaining = episodes
    chunk = min(check_interval, episodes)
    chunk_idx = 0
    while remaining > 0:
        run_n = min(chunk, remaining)
        if workers <= 1:
            part = _arena_worker(
                cand_path,
                best_path,
                run_n,
                seed + chunk_idx,
                device,
                reward_scale,
                chunk_idx,
            )
            diff, w, l, d, inv = part
            score_vals.extend(diff)
            wins += w
            losses += l
            draws += d
            invalid += inv
        else:
            per_worker = [run_n // workers] * workers
            for i in range(run_n % workers):
                per_worker[i] += 1
            futures = []
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                for i, n in enumerate(per_worker):
                    futures.append(
                        executor.submit(
                            _arena_worker,
                            cand_path,
                            best_path,
                            n,
                            seed + chunk_idx * 100 + i,
                            device,
                            reward_scale,
                            chunk_idx * 7 + i,
                        )
                    )
                for fut in as_completed(futures):
                    diff, w, l, d, inv = fut.result()
                    score_vals.extend(diff)
                    wins += w
                    losses += l
                    draws += d
                    invalid += inv

        remaining -= run_n
        chunk_idx += 1

        mean_score, ci_low, ci_high = mean_ci(score_vals, z)
        total = wins + losses
        win_rate = wins / total if total > 0 else 0.0
        _, win_low, win_high = wilson_ci(wins, total, z)

        log(
            "arena progress: n=%d mean=%.4f ci=[%.4f,%.4f] win=%.3f ci=[%.3f,%.3f]"
            % (len(score_vals), mean_score, ci_low, ci_high, win_rate, win_low, win_high)
        )

        if len(score_vals) >= min_episodes:
            accept_cfg = cfg["accept"]
            if accept_cfg["metric"] == "score":
                if ci_low > accept_cfg["score_threshold"]:
                    log("arena early accept: score ci_low > threshold")
                    break
                if ci_high < accept_cfg["score_threshold"]:
                    log("arena early reject: score ci_high < threshold")
                    break
            else:
                if win_low > accept_cfg["winrate_lower_bound"] and win_rate > accept_cfg["winrate_threshold"]:
                    log("arena early accept: winrate ci_low > bound")
                    break
                if win_high < accept_cfg["winrate_threshold"]:
                    log("arena early reject: winrate ci_high < threshold")
                    break

    mean_score, ci_low, ci_high = mean_ci(score_vals, z)
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0
    _, win_low, win_high = wilson_ci(wins, total, z)

    result = ArenaResult(
        score_mean=mean_score,
        score_ci_low=ci_low,
        score_ci_high=ci_high,
        win_rate=win_rate,
        win_ci_low=win_low,
        win_ci_high=win_high,
        wins=wins,
        losses=losses,
        draws=draws,
        invalid=invalid,
        episodes=len(score_vals),
    )
    return result


def decide_accept(cfg: Dict[str, Any], result: ArenaResult) -> Tuple[bool, str]:
    accept_cfg = cfg["accept"]
    if accept_cfg["metric"] == "score":
        if result.score_ci_low > accept_cfg["score_threshold"]:
            return True, "score_ci_low %.4f > %.4f" % (result.score_ci_low, accept_cfg["score_threshold"])
        return False, "score_ci_low %.4f <= %.4f" % (result.score_ci_low, accept_cfg["score_threshold"])
    if result.win_ci_low > accept_cfg["winrate_lower_bound"] and result.win_rate > accept_cfg["winrate_threshold"]:
        return True, "win_ci_low %.3f > %.3f" % (result.win_ci_low, accept_cfg["winrate_lower_bound"])
    return False, "win_ci_low %.3f <= %.3f" % (result.win_ci_low, accept_cfg["winrate_lower_bound"])


def cleanup(paths: Dict[str, str], cfg: Dict[str, Any], keep_iters: int, keep_models: int) -> None:
    data_dir = paths["data"]
    all_iters = sorted(
        [d for d in os.listdir(data_dir) if d.startswith("iter_")],
        reverse=True,
    )
    for d in all_iters[keep_iters:]:
        shutil.rmtree(os.path.join(data_dir, d), ignore_errors=True)

    mix_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith("mix_iter_")])
    for d in mix_dirs:
        shutil.rmtree(os.path.join(data_dir, d), ignore_errors=True)

    cand_dir = paths["candidates"]
    candidates = sorted(
        [p for p in os.listdir(cand_dir) if p.endswith(".pt")],
        reverse=True,
    )
    for p in candidates[keep_models:]:
        os.remove(os.path.join(cand_dir, p))


def apply_dry_run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deep_update(cfg, {})
    cfg["max_iters"] = 1
    cfg["self_play"]["episodes"] = 4
    cfg["self_play"]["simulations"] = 8
    cfg["self_play"]["save_every"] = 256
    cfg["self_play"]["workers"] = 1
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 128
    cfg["train"]["steps_per_epoch"] = 10
    cfg["arena"]["episodes"] = 6
    cfg["arena"]["min_episodes"] = 4
    cfg["arena"]["check_interval"] = 2
    cfg["arena"]["workers"] = 1
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero training loop for oracle teacher")
    parser.add_argument("--run_name", required=True, help="Run name under runs/az/")
    parser.add_argument("--config", default="", help="Optional json/yaml config file")
    parser.add_argument("--init_model", default="", help="Initial best model checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume if state.json exists")
    parser.add_argument("--max_iters", type=int, default=None, help="Override max iters")
    parser.add_argument("--dry_run", action="store_true", help="Run a small smoke test")
    args = parser.parse_args()

    cfg = default_config()
    cfg = deep_update(cfg, load_config(args.config))
    if args.max_iters is not None:
        cfg["max_iters"] = args.max_iters
    if args.dry_run:
        cfg = apply_dry_run(cfg)

    paths = prepare_run_dir(args.run_name)
    if os.path.exists(paths["state"]) and not args.resume:
        raise RuntimeError("state exists, use --resume to continue")

    state = load_or_init_state(paths, cfg, args.init_model or None)
    start_iter = state["iter"] + 1
    max_iters = int(cfg["max_iters"])
    log("az_loop start: run=%s start_iter=%d max_iters=%d" % (args.run_name, start_iter, max_iters))

    for iter_id in range(start_iter, max_iters + 1):
        log("===== iter %04d =====" % iter_id)
        try:
            sp_temp = cfg["self_play"]["temperature"]
            if iter_id >= int(cfg["self_play"]["temperature_drop_iter"]):
                sp_temp = 0.0
            cfg["self_play"]["temperature"] = sp_temp

            iter_dir = spawn_self_play(cfg, paths, iter_id, state["best_model"])
            data_iters = list(range(max(1, iter_id - cfg["data_window"] + 1), iter_id + 1))
            mix_dir = build_mix_dir(paths, iter_id, data_iters, cfg["train"].get("replay_buffer_samples"))
            cand_path = train_candidate(cfg, paths, iter_id, mix_dir)
            shutil.rmtree(mix_dir, ignore_errors=True)

            result = run_arena(cfg, paths, iter_id, state["best_model"], cand_path)
            arena_path = os.path.join(paths["arena"], "iter_%04d.json" % iter_id)
            safe_write_json(
                arena_path,
                {
                    "iter": iter_id,
                    "cand": cand_path,
                    "best": state["best_model"],
                    "score_mean": result.score_mean,
                    "score_ci": [result.score_ci_low, result.score_ci_high],
                    "win_rate": result.win_rate,
                    "win_ci": [result.win_ci_low, result.win_ci_high],
                    "wins": result.wins,
                    "losses": result.losses,
                    "draws": result.draws,
                    "invalid": result.invalid,
                    "episodes": result.episodes,
                },
            )

            accept, reason = decide_accept(cfg, result)
            if accept:
                shutil.copy2(cand_path, paths["best"])
                state["best_model"] = paths["best"]
                log("accept: %s" % reason)
            else:
                log("reject: %s" % reason)

            state["iter"] = iter_id
            state["last_arena"] = arena_path
            state["last_update"] = _now()
            safe_write_json(paths["state"], state)

            cleanup(paths, cfg, cfg["cleanup"]["keep_data_iters"], cfg["cleanup"]["keep_models"])
        except Exception as exc:
            err_path = os.path.join(paths["logs"], "iter_%04d_error.txt" % iter_id)
            with open(err_path, "w", encoding="utf-8") as f:
                f.write("[%s] %s\n" % (_now(), str(exc)))
                f.write(traceback.format_exc())
            log("error: %s (see %s)" % (exc, err_path))
            state["last_update"] = _now()
            safe_write_json(paths["state"], state)
            raise


if __name__ == "__main__":
    main()
