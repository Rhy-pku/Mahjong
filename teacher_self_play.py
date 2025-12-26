#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-play evaluation for oracle teacher models.
"""

import argparse
import random
from collections import defaultdict

import numpy as np
import torch

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
    for key in ("pre_conv.0.weight", "pre_conv.0.weight"):
        if key in state_dict:
            return state_dict[key].shape[1]
    raise KeyError("pre_conv.0.weight not found in state_dict")


def infer_hidden_dim(state_dict, fallback):
    key = "fusion.0.weight"
    if key in state_dict:
        return state_dict[key].shape[0]
    return fallback


def select_actions(model, device, env, obs_dict, in_channels):
    action_dict = {}
    batch_obs = []
    batch_masks = []
    batch_players = []
    batch_names = []

    for name, obs in obs_dict.items():
        player = int(name.split("_")[-1]) - 1
        batch_players.append(player)
        batch_names.append(name)
        batch_obs.append(build_oracle_obs(env, player, obs["observation"], in_channels))
        batch_masks.append(obs["action_mask"])

    if batch_obs:
        obs_t = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device)
        mask_t = torch.tensor(np.stack(batch_masks), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = model(obs_t)
        masked_logits = _apply_action_mask(logits, mask_t)
        actions = torch.argmax(masked_logits, dim=1).cpu().numpy().tolist()
        for name, action in zip(batch_names, actions):
            action_dict[name] = int(action)

    return action_dict


def evaluate(model, device, episodes, duplicate, seed, parallel_envs, use_amp):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    n_envs = min(max(1, parallel_envs), episodes)
    envs = []
    for _ in range(n_envs):
        env = MahjongGBEnv(config={"agent_clz": FeatureAgent, "duplicate": duplicate})
        envs.append({"env": env, "obs": env.reset()})

    agent_names = envs[0]["env"].agent_names
    total_scores = {name: 0.0 for name in agent_names}
    win_counts = {name: 0 for name in agent_names}
    draw_count = 0
    invalid_count = 0

    in_channels = model.pre_conv[0].weight.shape[1]

    finished = 0
    while finished < episodes:
        batch_obs = []
        batch_masks = []
        batch_meta = []
        for env_idx, slot in enumerate(envs):
            if slot is None:
                continue
            obs_dict = slot["obs"]
            for name, obs in obs_dict.items():
                player = int(name.split("_")[-1]) - 1
                batch_meta.append((env_idx, name))
                batch_obs.append(build_oracle_obs(slot["env"], player, obs["observation"], in_channels))
                batch_masks.append(obs["action_mask"])

        if not batch_obs:
            break

        obs_t = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device)
        mask_t = torch.tensor(np.stack(batch_masks), dtype=torch.float32, device=device)
        with torch.inference_mode():
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits, _ = model(obs_t)
            else:
                logits, _ = model(obs_t)
        masked_logits = _apply_action_mask(logits, mask_t)
        actions = torch.argmax(masked_logits, dim=1).cpu().numpy().tolist()

        action_dicts = []
        for slot in envs:
            if slot is None:
                action_dicts.append(None)
            else:
                action_dicts.append({name: 0 for name in agent_names})
        for (env_idx, name), action in zip(batch_meta, actions):
            action_dicts[env_idx][name] = int(action)

        for env_idx, slot in enumerate(envs):
            if slot is None:
                continue
            obs, rewards, done = slot["env"].step(action_dicts[env_idx])
            if done:
                for name in agent_names:
                    total_scores[name] += rewards.get(name, 0)

                reward_vals = [rewards.get(name, 0) for name in agent_names]
                if any(r == -30 for r in reward_vals):
                    invalid_count += 1
                elif all(r == 0 for r in reward_vals):
                    draw_count += 1
                else:
                    max_reward = max(reward_vals)
                    if reward_vals.count(max_reward) == 1:
                        winner = agent_names[reward_vals.index(max_reward)]
                        win_counts[winner] += 1

                finished += 1
                if finished < episodes:
                    obs = slot["env"].reset()
                else:
                    envs[env_idx] = None
                    continue
            slot["obs"] = obs

    win_rates = {name: win_counts[name] / episodes for name in agent_names}
    return total_scores, win_rates, draw_count, invalid_count


def main():
    parser = argparse.ArgumentParser(description="Teacher self-play evaluation")
    parser.add_argument("--model", required=True, help="Teacher checkpoint path")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dim fallback")
    parser.add_argument("--duplicate", action="store_true", help="Use duplicated wall env")
    parser.add_argument("--parallel_envs", type=int, default=1, help="Number of envs to run in parallel")
    parser.add_argument("--amp", action="store_true", help="Enable AMP inference on cuda")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    device = torch.device(args.device)
    state_dict = load_state_dict_compat(args.model)
    in_channels = infer_in_channels(state_dict)
    hidden_dim = infer_hidden_dim(state_dict, args.hidden_dim)
    model = PretrainModel(hidden_dim=hidden_dim, use_vec=False, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    total_scores, win_rates, draw_count, invalid_count = evaluate(
        model=model,
        device=device,
        episodes=args.episodes,
        duplicate=args.duplicate,
        seed=args.seed,
        parallel_envs=args.parallel_envs,
        use_amp=args.amp,
    )

    print("episodes:", args.episodes)
    print("draws:", draw_count, "invalid:", invalid_count)
    print("total_scores:", total_scores)
    print("win_rates:", win_rates)


if __name__ == "__main__":
    main()
