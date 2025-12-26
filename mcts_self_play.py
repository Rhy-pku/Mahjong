#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MCTS self-play for oracle teacher.
"""

import argparse
import copy
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from env import MahjongGBEnv
from feature_10m import FeatureAgent10M as FeatureAgent
from model_pretrain import PretrainModel

TILE_LIST = [
    *("W%d" % (i + 1) for i in range(9)),
    *("T%d" % (i + 1) for i in range(9)),
    *("B%d" % (i + 1) for i in range(9)),
    *("F%d" % (i + 1) for i in range(4)),
    *("J%d" % (i + 1) for i in range(3)),
]
OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}
STUDENT_CHANNELS = 60


def _tile_rc(tile):
    idx = OFFSET_TILE.get(tile, -1)
    if idx < 0:
        return None
    return divmod(idx, 9)


def _encode_hand(obs, base, tiles):
    for tile in tiles:
        rc = _tile_rc(tile)
        if rc is None:
            continue
        r, c = rc
        for i in range(4):
            if obs[base + i, r, c] == 0:
                obs[base + i, r, c] = 1
                break


def _get_hand_with_draw(env, pid):
    hand = list(env.hands[pid])
    if getattr(env, "state", None) == 1 and env.curPlayer == pid and env.curTile is not None:
        if env.curTile not in hand:
            hand.append(env.curTile)
    return hand


def _remaining_counts(env):
    counts = defaultdict(int)
    wall = env.tileWall
    if wall and isinstance(wall[0], list):
        for sub in wall:
            for tile in sub:
                counts[tile] += 1
    else:
        for tile in wall:
            counts[tile] += 1
    return counts


def build_oracle_obs(env, player, student_obs, in_channels):
    extra_hand_channels = 12
    extra_remaining_channels = in_channels - (STUDENT_CHANNELS + extra_hand_channels)
    if extra_remaining_channels not in (0, 1, 34):
        raise ValueError("Unsupported oracle channels: %d" % in_channels)

    out = np.zeros((in_channels, 4, 9), dtype=np.int8)
    out[:STUDENT_CHANNELS] = student_obs

    order = [player, (player + 1) % 4, (player + 2) % 4, (player + 3) % 4]
    offset = STUDENT_CHANNELS
    for pid in order[1:]:
        _encode_hand(out, offset, _get_hand_with_draw(env, pid))
        offset += 4

    if extra_remaining_channels > 0:
        counts = _remaining_counts(env)
        if extra_remaining_channels == 1:
            for tile_idx, tile in enumerate(TILE_LIST):
                r, c = divmod(tile_idx, 9)
                out[offset, r, c] = min(counts.get(tile, 0), 4)
        else:
            for tile_idx, tile in enumerate(TILE_LIST):
                r, c = divmod(tile_idx, 9)
                out[offset + tile_idx, r, c] = min(counts.get(tile, 0), 4)

    return out


def _apply_action_mask(logits, mask):
    inf_mask = torch.clamp(torch.log(mask + 1e-45), min=-1e38, max=0)
    return logits + inf_mask


def load_state_dict_compat(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def infer_in_channels(state_dict):
    if "pre_conv.0.weight" in state_dict:
        return state_dict["pre_conv.0.weight"].shape[1]
    raise KeyError("pre_conv.0.weight not found in state_dict")


def infer_hidden_dim(state_dict, fallback):
    if "fusion.0.weight" in state_dict:
        return state_dict["fusion.0.weight"].shape[0]
    return fallback


def infer_value_dim(state_dict):
    key = "value_head.2.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    return 1


def _player_from_name(name):
    return int(name.split("_")[-1]) - 1


def policy_actions(model, device, env, obs_dict, in_channels):
    action_dict = {}
    batch_obs = []
    batch_masks = []
    batch_names = []

    for name, obs in obs_dict.items():
        player = _player_from_name(name)
        batch_names.append(name)
        batch_obs.append(build_oracle_obs(env, player, obs["observation"], in_channels))
        batch_masks.append(obs["action_mask"])

    if batch_obs:
        obs_t = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device)
        mask_t = torch.tensor(np.stack(batch_masks), dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits, _ = model(obs_t)
        masked_logits = _apply_action_mask(logits, mask_t)
        actions = torch.argmax(masked_logits, dim=1).cpu().numpy().tolist()
        for name, action in zip(batch_names, actions):
            action_dict[name] = int(action)
    return action_dict


def policy_value(model, device, env, player, obs, in_channels, value_dim):
    oracle_obs = build_oracle_obs(env, player, obs["observation"], in_channels)
    obs_t = torch.tensor(oracle_obs[None, ...], dtype=torch.float32, device=device)
    mask_t = torch.tensor(obs["action_mask"][None, ...], dtype=torch.float32, device=device)
    with torch.inference_mode():
        logits, value = model(obs_t)
    masked_logits = _apply_action_mask(logits, mask_t)
    probs = torch.softmax(masked_logits, dim=1).cpu().numpy()[0]
    if value_dim == 1:
        value_vec = np.zeros(4, dtype=np.float32)
        value_vec[player] = float(value.cpu().numpy()[0, 0])
    else:
        value_vec = value.cpu().numpy()[0].astype(np.float32)
    return probs, value_vec


def reward_to_vec(env, rewards, reward_scale):
    if rewards:
        vals = [rewards.get(name, 0) for name in env.agent_names]
        return np.array(vals, dtype=np.float32) / reward_scale
    if getattr(env, "reward", None) is not None:
        return np.array(env.reward, dtype=np.float32) / reward_scale
    return np.zeros(4, dtype=np.float32)


def _shuffle_wall(env, rng):
    wall = env.tileWall
    if wall and isinstance(wall[0], list):
        for sub in wall:
            rng.shuffle(sub)
    else:
        rng.shuffle(wall)


class MCTSNode:
    def __init__(self, action_size, value_dim):
        self.action_size = action_size
        self.value_dim = value_dim
        self.prior = None
        self.valid_actions = None
        self.n_visits = 0
        self.nsa = np.zeros(action_size, dtype=np.int32)
        self.wsa = np.zeros((action_size, value_dim), dtype=np.float32)
        self.children = {}
        self.expanded = False

    def expand(self, action_mask, priors):
        valid = np.flatnonzero(action_mask > 0)
        if valid.size == 0:
            valid = np.arange(self.action_size, dtype=np.int32)
        probs = np.zeros_like(priors, dtype=np.float32)
        if priors.sum() <= 1e-8:
            probs[valid] = 1.0 / len(valid)
        else:
            probs = priors.astype(np.float32)
            probs[action_mask <= 0] = 0.0
            total = probs.sum()
            if total <= 1e-8:
                probs[valid] = 1.0 / len(valid)
            else:
                probs /= total
        self.prior = probs
        self.valid_actions = valid
        self.expanded = True

    def select(self, player, c_puct):
        n = max(self.n_visits, 1)
        valid = self.valid_actions
        q = np.zeros(self.action_size, dtype=np.float32)
        nsa = self.nsa[valid].astype(np.float32)
        wsa = self.wsa[valid, player]
        q_vals = wsa / np.maximum(nsa, 1.0)
        u = c_puct * self.prior[valid] * math.sqrt(n) / (1.0 + nsa)
        scores = q_vals + u
        return int(valid[int(np.argmax(scores))])


def simulate(
    env,
    obs_dict,
    root,
    model,
    device,
    in_channels,
    value_dim,
    c_puct,
    reward_scale,
):
    path = []
    node = root
    value_vec = None
    while True:
        if env.done:
            value_vec = reward_to_vec(env, None, reward_scale)
            break
        if len(obs_dict) != 1:
            action_dict = {name: 0 for name in env.agent_names}
            action_dict.update(policy_actions(model, device, env, obs_dict, in_channels))
            obs_dict, rewards, done = env.step(action_dict)
            if done:
                value_vec = reward_to_vec(env, rewards, reward_scale)
                break
            continue

        name, obs = next(iter(obs_dict.items()))
        player = _player_from_name(name)
        if not node.expanded:
            priors, value_vec = policy_value(model, device, env, player, obs, in_channels, value_dim)
            node.expand(obs["action_mask"], priors)
            break

        action = node.select(player, c_puct)
        path.append((node, action))
        child = node.children.get(action)
        if child is None:
            child = MCTSNode(node.action_size, node.value_dim)
            node.children[action] = child
        action_dict = {n: 0 for n in env.agent_names}
        action_dict[name] = action
        obs_dict, rewards, done = env.step(action_dict)
        if done:
            value_vec = reward_to_vec(env, rewards, reward_scale)
            break
        node = child

    if value_vec is None:
        value_vec = np.zeros(4, dtype=np.float32)

    if path:
        for n, a in path:
            n.n_visits += 1
            n.nsa[a] += 1
            n.wsa[a] += value_vec
    else:
        root.n_visits += 1
    return value_vec


def mcts_action(
    env,
    obs_dict,
    model,
    device,
    in_channels,
    value_dim,
    simulations,
    c_puct,
    reward_scale,
    determinize,
    rng,
    temperature,
):
    name, obs = next(iter(obs_dict.items()))
    action_size = obs["action_mask"].shape[0]
    root = MCTSNode(action_size, value_dim)
    for _ in range(simulations):
        env_copy = copy.deepcopy(env)
        if determinize:
            _shuffle_wall(env_copy, rng)
        obs_copy = env_copy._obs()
        simulate(
            env_copy,
            obs_copy,
            root,
            model,
            device,
            in_channels,
            value_dim,
            c_puct,
            reward_scale,
        )

    if not root.expanded:
        player = _player_from_name(name)
        priors, _ = policy_value(model, device, env, player, obs, in_channels, value_dim)
        root.expand(obs["action_mask"], priors)

    counts = root.nsa.astype(np.float32)
    mask = obs["action_mask"].astype(np.float32)
    counts = counts * mask
    valid = root.valid_actions
    if valid is None or valid.size == 0:
        pi = mask
        if pi.sum() > 0:
            pi = pi / pi.sum()
        else:
            pi = np.ones_like(counts) / len(counts)
        return int(np.argmax(counts)), pi
    if temperature <= 1e-6:
        if counts.sum() <= 1e-8:
            return int(valid[int(np.argmax(mask[valid]))]), mask / max(mask.sum(), 1.0)
        pi = counts / max(counts.sum(), 1.0)
        return int(valid[int(np.argmax(counts[valid]))]), pi
    probs = counts.copy()
    probs[probs < 0] = 0
    probs = probs ** (1.0 / temperature)
    if probs.sum() <= 1e-8:
        pi = mask
        if pi.sum() > 0:
            pi = pi / pi.sum()
        else:
            pi = np.ones_like(counts) / len(counts)
        return int(valid[int(np.argmax(counts[valid]))]), pi
    probs /= probs.sum()
    return int(np.random.choice(np.arange(len(probs)), p=probs)), probs


def _save_batch(out_dir, file_idx, samples):
    out_path = os.path.join(out_dir, f"{file_idx}.npz")
    np.savez(
        out_path,
        oracle_obs=np.stack([s["oracle_obs"] for s in samples]).astype(np.int8),
        action_mask=np.stack([s["action_mask"] for s in samples]).astype(np.int8),
        pi=np.stack([s["pi"] for s in samples]).astype(np.float32),
        player=np.array([s["player"] for s in samples], dtype=np.int8),
        reward_vec=np.stack([s["reward_vec"] for s in samples]).astype(np.float32),
    )
    print("saved samples:", out_path, "count:", len(samples))


def self_play(
    model,
    device,
    episodes,
    duplicate,
    seed,
    simulations,
    c_puct,
    reward_scale,
    determinize,
    temperature,
    out_dir,
    save_every,
    log_interval,
    wandb_run=None,
    wandb_log_interval=10,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    env = MahjongGBEnv(config={"agent_clz": FeatureAgent, "duplicate": duplicate})
    agent_names = env.agent_names
    total_scores = {name: 0.0 for name in agent_names}
    win_counts = {name: 0 for name in agent_names}
    draw_count = 0
    invalid_count = 0

    in_channels = model.pre_conv[0].weight.shape[1]
    value_dim = model.value_head[-1].out_features
    rng = random.Random(seed)
    samples = []
    file_idx = 0
    total_samples = 0
    start_time = time.time()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(
        "self-play config: episodes=%d sims=%d c_puct=%.3f determinize=%s temp=%.3f out_dir=%s"
        % (episodes, simulations, c_puct, str(determinize), temperature, out_dir or "None")
    )
    pbar = None
    if tqdm is not None:
        pbar = tqdm(range(episodes), unit="ep", dynamic_ncols=True)
        episode_iter = pbar
    else:
        episode_iter = range(episodes)

    for episode_idx in episode_iter:
        obs = env.reset()
        done = False
        rewards = None
        episode_samples = []
        while not done:
            if len(obs) == 1:
                action, pi = mcts_action(
                    env=env,
                    obs_dict=obs,
                    model=model,
                    device=device,
                    in_channels=in_channels,
                    value_dim=value_dim,
                    simulations=simulations,
                    c_puct=c_puct,
                    reward_scale=reward_scale,
                    determinize=determinize,
                    rng=rng,
                    temperature=temperature,
                )
                name = next(iter(obs.keys()))
                player = _player_from_name(name)
                episode_samples.append(
                    {
                        "oracle_obs": build_oracle_obs(env, player, obs[name]["observation"], in_channels),
                        "action_mask": obs[name]["action_mask"],
                        "pi": pi,
                        "player": player,
                    }
                )
                action_dict = {n: 0 for n in agent_names}
                action_dict[name] = int(action)
            else:
                action_dict = {n: 0 for n in agent_names}
                action_dict.update(policy_actions(model, device, env, obs, in_channels))
            obs, rewards, done = env.step(action_dict)

        for name in agent_names:
            total_scores[name] += rewards.get(name, 0)

        reward_vals = [rewards.get(name, 0) for name in agent_names]
        if any(r == -30 for r in reward_vals):
            invalid_count += 1
            continue
        if all(r == 0 for r in reward_vals):
            draw_count += 1
            continue
        max_reward = max(reward_vals)
        if reward_vals.count(max_reward) == 1:
            winner = agent_names[reward_vals.index(max_reward)]
            win_counts[winner] += 1

        if episode_samples:
            reward_vec = reward_to_vec(env, rewards, reward_scale)
            for s in episode_samples:
                s["reward_vec"] = reward_vec
            if out_dir:
                samples.extend(episode_samples)
                total_samples += len(episode_samples)
                if len(samples) >= save_every:
                    _save_batch(out_dir, file_idx, samples)
                    file_idx += 1
                    samples = []

        if log_interval and ((episode_idx + 1) % log_interval == 0):
            elapsed = time.time() - start_time
            avg_ep_time = elapsed / max(episode_idx + 1, 1)
            print(
                "episode %d/%d steps=%d samples=%d wins=%s draws=%d invalid=%d avg_ep=%.2fs"
                % (
                    episode_idx + 1,
                    episodes,
                    len(episode_samples),
                    total_samples,
                    win_counts,
                    draw_count,
                    invalid_count,
                    avg_ep_time,
                )
            )

        if wandb_run and ((episode_idx + 1) % wandb_log_interval == 0):
            win_rates = {name: win_counts[name] / max(episode_idx + 1, 1) for name in agent_names}
            wandb_run.log(
                {
                    "episode": episode_idx + 1,
                    "episode/steps": len(episode_samples),
                    "episode/samples_total": total_samples,
                    "episode/draws": draw_count,
                    "episode/invalid": invalid_count,
                    "win_rate/player_1": win_rates.get("player_1", 0.0),
                    "win_rate/player_2": win_rates.get("player_2", 0.0),
                    "win_rate/player_3": win_rates.get("player_3", 0.0),
                    "win_rate/player_4": win_rates.get("player_4", 0.0),
                }
            )

        if pbar is not None:
            pbar.set_postfix(samples=total_samples, draws=draw_count, invalid=invalid_count)

    win_rates = {name: win_counts[name] / episodes for name in agent_names}
    if out_dir and samples:
        _save_batch(out_dir, file_idx, samples)
    if pbar is not None:
        pbar.close()
    return total_scores, win_rates, draw_count, invalid_count


def main():
    parser = argparse.ArgumentParser(description="Minimal MCTS self-play for oracle teacher")
    parser.add_argument("--model", required=True, help="Teacher checkpoint path")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dim fallback")
    parser.add_argument("--duplicate", action="store_true", help="Use duplicated wall env")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--simulations", type=int, default=64, help="MCTS sims per decision")
    parser.add_argument("--c_puct", type=float, default=1.5, help="PUCT constant")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Divide rewards by this value")
    parser.add_argument("--determinize", action="store_true", help="Shuffle wall per simulation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Action temperature")
    parser.add_argument("--out_dir", default="", help="Save self-play samples to this dir")
    parser.add_argument("--save_every", type=int, default=4096, help="Samples per npz")
    parser.add_argument("--log_interval", type=int, default=1, help="Print every N episodes")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="mahjong-az", help="W&B project name")
    parser.add_argument("--wandb_run_name", default="", help="W&B run name")
    parser.add_argument("--wandb_log_interval", type=int, default=10, help="Log every N episodes")
    args = parser.parse_args()

    device = torch.device(args.device)
    state_dict = load_state_dict_compat(args.model)
    in_channels = infer_in_channels(state_dict)
    hidden_dim = infer_hidden_dim(state_dict, args.hidden_dim)
    value_dim = infer_value_dim(state_dict)
    model = PretrainModel(
        hidden_dim=hidden_dim,
        use_vec=False,
        in_channels=in_channels,
        value_dim=value_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config={
                "episodes": args.episodes,
                "simulations": args.simulations,
                "c_puct": args.c_puct,
                "reward_scale": args.reward_scale,
                "determinize": args.determinize,
                "temperature": args.temperature,
            },
        )

    total_scores, win_rates, draw_count, invalid_count = self_play(
        model=model,
        device=device,
        episodes=args.episodes,
        duplicate=args.duplicate,
        seed=args.seed,
        simulations=args.simulations,
        c_puct=args.c_puct,
        reward_scale=args.reward_scale,
        determinize=args.determinize,
        temperature=args.temperature,
        out_dir=args.out_dir.strip() or None,
        save_every=args.save_every,
        log_interval=args.log_interval,
        wandb_run=wandb_run,
        wandb_log_interval=args.wandb_log_interval,
    )

    print("episodes:", args.episodes)
    print("draws:", draw_count, "invalid:", invalid_count)
    print("total_scores:", total_scores)
    print("win_rates:", win_rates)
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
