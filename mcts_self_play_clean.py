#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal MCTS self-play for oracle teacher.
"""

import argparse
import copy
import json
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
DRAW_PENALTY = -0.05


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


def _remaining_wall_tiles(env):
    wall = getattr(env, "tileWall", None)
    if not wall:
        return 0
    if isinstance(wall, list) and wall and isinstance(wall[0], list):
        return sum(len(sub) for sub in wall)
    return len(wall)


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


def _get_hu_action(mask):
    if mask.shape[0] == 0:
        return None
    if mask.shape[0] > 1 and mask[1] > 0:
        return 1
    return None


def _candidate_actions(prior, valid, n_visits, top_k, min_actions, policy_mass):
    if valid is None or len(valid) == 0:
        return valid
    pri = prior[valid]
    order = valid[np.argsort(-pri)]
    k_mass = len(order)
    if 0.0 < policy_mass < 1.0:
        cum = np.cumsum(pri[np.argsort(-pri)])
        k_mass = int(np.searchsorted(cum, policy_mass) + 1)
    k_mass = max(k_mass, min_actions)
    if top_k and top_k > 0:
        k_pw = max(min_actions, int(2 + math.sqrt(max(n_visits, 1))))
        k_pw = min(k_pw, top_k)
        k_keep = min(k_mass, k_pw)
    else:
        k_keep = k_mass
    k_keep = min(k_keep, len(order))
    return order[:k_keep]


def _safe_env_step(env, action_dict):
    try:
        obs, rewards, done = env.step(action_dict)
        return obs, rewards, done, False
    except Exception:
        return {}, None, True, True


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
        obs_t = _to_device(np.stack(batch_obs), device)
        mask_t = _to_device(np.stack(batch_masks), device)
        with torch.inference_mode():
            logits, _ = model(obs_t)
        masked_logits = _apply_action_mask(logits, mask_t)
        actions = torch.argmax(masked_logits, dim=1).cpu().numpy().tolist()
        for name, action in zip(batch_names, actions):
            action_dict[name] = int(action)
    return action_dict


def _parse_mcts_players(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                data = json.loads(s)
                return [int(x) for x in data]
            except Exception:
                return []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [int(p) for p in parts]
    return [int(raw)]


def _normalize_mcts_players(players):
    uniq = []
    for p in players:
        if p < 0 or p > 3:
            raise ValueError("mcts_players must be in [0, 3], got %s" % p)
        if p not in uniq:
            uniq.append(p)
    return uniq


def policy_action_single(model, device, env, player, obs, in_channels):
    oracle_obs = build_oracle_obs(env, player, obs["observation"], in_channels)
    obs_t = _to_device(oracle_obs[None, ...], device)
    mask_t = _to_device(obs["action_mask"][None, ...], device)
    with torch.inference_mode():
        logits, _ = model(obs_t)
    masked_logits = _apply_action_mask(logits, mask_t)
    return int(torch.argmax(masked_logits, dim=1).item())


def policy_value(model, device, env, player, obs, in_channels, value_dim, value_scale=1.0):
    oracle_obs = build_oracle_obs(env, player, obs["observation"], in_channels)
    obs_t = _to_device(oracle_obs[None, ...], device)
    mask_t = _to_device(obs["action_mask"][None, ...], device)
    with torch.inference_mode():
        logits, value = model(obs_t)
    masked_logits = _apply_action_mask(logits, mask_t)
    probs = torch.softmax(masked_logits, dim=1).cpu().numpy()[0]
    if value_dim == 1:
        value_vec = np.zeros(4, dtype=np.float32)
        value_vec[player] = float(value.cpu().numpy()[0, 0]) * float(value_scale)
    else:
        value_vec = value.cpu().numpy()[0].astype(np.float32) * float(value_scale)
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


def _clone_env(env):
    if hasattr(env, "fast_clone"):
        return env.fast_clone()
    return copy.deepcopy(env)


def _to_device(array, device):
    if device.type == "cuda":
        tensor = torch.as_tensor(array, dtype=torch.float32).contiguous()
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:
            pass
        return tensor.to(device, non_blocking=True)
    return torch.as_tensor(array, dtype=torch.float32, device=device).contiguous()


class MinMaxStats:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        if value is None:
            return
        if not math.isfinite(value):
            return
        if value > self.maximum:
            self.maximum = value
        if value < self.minimum:
            self.minimum = value

    def normalize(self, values):
        if self.maximum > self.minimum:
            return (values - self.minimum) / (self.maximum - self.minimum)
        return values


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
        self.player = None

    def expand(self, action_mask, priors):
        valid = np.flatnonzero(action_mask > 0)
        if valid.size == 0:
            valid = np.arange(self.action_size, dtype=np.int32)
        probs = priors.astype(np.float32)
        probs[action_mask <= 0] = 0.0
        total = probs.sum()
        if total <= 1e-8:
            probs[:] = 0.0
            probs[valid] = 1.0 / len(valid)
        else:
            probs /= total
        self.prior = probs
        self.valid_actions = valid
        self.expanded = True

    def select(self, player, c_puct, top_k, min_actions, policy_mass, min_max_stats=None):
        n = max(self.n_visits, 1)
        valid = _candidate_actions(self.prior, self.valid_actions, self.n_visits, top_k, min_actions, policy_mass)
        q = np.zeros(self.action_size, dtype=np.float32)
        nsa = self.nsa[valid].astype(np.float32)
        wsa = self.wsa[valid, player]
        q_vals = wsa / np.maximum(nsa, 1.0)
        if min_max_stats is not None:
            q_vals = min_max_stats.normalize(q_vals)
        u = c_puct * self.prior[valid] * math.sqrt(n) / (1.0 + nsa)
        scores = q_vals + u
        return int(valid[int(np.argmax(scores))])


def _backprop(path, root, value_vec, min_max_stats=None):
    if path:
        for node, action in path:
            node.n_visits += 1
            node.nsa[action] += 1
            node.wsa[action] += value_vec
            if min_max_stats is not None and node.player is not None:
                q_val = node.wsa[action, node.player] / node.nsa[action]
                min_max_stats.update(float(q_val))
    else:
        root.n_visits += 1


def _simulate_to_leaf(
    env,
    obs_dict,
    root,
    model,
    device,
    in_channels,
    value_dim,
    c_puct,
    reward_scale,
    top_k,
    min_actions,
    policy_mass,
    mcts_player_ids,
    min_max_stats=None,
):
    path = []
    node = root
    while True:
        if env.done:
            value_vec = reward_to_vec(env, None, reward_scale)
            return "terminal", (value_vec, path)
        if len(obs_dict) != 1:
            action_dict = {name: 0 for name in env.agent_names}
            action_dict.update(policy_actions(model, device, env, obs_dict, in_channels))
            obs_dict, rewards, done, err = _safe_env_step(env, action_dict)
            if done:
                if err:
                    value_vec = np.zeros(4, dtype=np.float32)
                else:
                    value_vec = reward_to_vec(env, rewards, reward_scale)
                return "terminal", (value_vec, path)
            continue

        name, obs = next(iter(obs_dict.items()))
        player = _player_from_name(name)
        node.player = player
        if not node.expanded:
            return "leaf", (node, path, player, obs, env)

        use_mcts = mcts_player_ids is None or player in mcts_player_ids
        if use_mcts:
            action = node.select(player, c_puct, top_k, min_actions, policy_mass, min_max_stats)
        else:
            action = int(np.argmax(node.prior))
        path.append((node, action))
        child = node.children.get(action)
        if child is None:
            child = MCTSNode(node.action_size, node.value_dim)
            node.children[action] = child
        action_dict = {n: 0 for n in env.agent_names}
        action_dict[name] = action
        obs_dict, rewards, done, err = _safe_env_step(env, action_dict)
        if done:
            if err:
                value_vec = np.zeros(4, dtype=np.float32)
            else:
                value_vec = reward_to_vec(env, rewards, reward_scale)
            return "terminal", (value_vec, path)
        node = child


def _eval_leaf_batch(
    leaf_batch,
    root,
    model,
    device,
    in_channels,
    value_dim,
    value_scale=1.0,
    min_max_stats=None,
):
    obs_list = []
    mask_list = []
    players = []
    nodes = []
    paths = []
    for node, path, player, obs, env in leaf_batch:
        obs_list.append(build_oracle_obs(env, player, obs["observation"], in_channels))
        mask_list.append(obs["action_mask"])
        players.append(player)
        nodes.append(node)
        paths.append(path)

    obs_t = _to_device(np.stack(obs_list), device)
    mask_t = _to_device(np.stack(mask_list), device)
    with torch.inference_mode():
        logits, values = model(obs_t)
    masked_logits = _apply_action_mask(logits, mask_t)
    probs = torch.softmax(masked_logits, dim=1).cpu().numpy()
    values_np = values.cpu().numpy()

    for i in range(len(leaf_batch)):
        priors = probs[i]
        nodes[i].player = players[i]
        nodes[i].expand(mask_list[i], priors)
        if value_dim == 1:
            value_vec = np.zeros(4, dtype=np.float32)
            value_vec[players[i]] = float(values_np[i, 0]) * float(value_scale)
        else:
            value_vec = values_np[i].astype(np.float32) * float(value_scale)
        _backprop(paths[i], root, value_vec, min_max_stats)


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
    top_k,
    min_actions,
    policy_mass,
    leaf_batch_size,
    mcts_player_ids=None,
    value_scale=1.0,
):
    name, obs = next(iter(obs_dict.items()))
    action_size = obs["action_mask"].shape[0]
    hu_action = _get_hu_action(obs["action_mask"])
    if hu_action is not None:
        pi = np.zeros(action_size, dtype=np.float32)
        pi[hu_action] = 1.0
        return hu_action, pi
    root = MCTSNode(action_size, value_dim)
    leaf_batch = []
    leaf_batch_size = max(1, int(leaf_batch_size))
    if mcts_player_ids is not None:
        mcts_player_ids = set(mcts_player_ids)
    min_max_stats = MinMaxStats()
    for _ in range(simulations):
        env_copy = _clone_env(env)
        if determinize:
            _shuffle_wall(env_copy, rng)
        obs_copy = env_copy._obs()
        kind, payload = _simulate_to_leaf(
            env_copy,
            obs_copy,
            root,
            model,
            device,
            in_channels,
            value_dim,
            c_puct,
            reward_scale,
            top_k,
            min_actions,
            policy_mass,
            mcts_player_ids,
            min_max_stats,
        )
        if kind == "terminal":
            value_vec, path = payload
            _backprop(path, root, value_vec, min_max_stats)
        else:
            leaf_batch.append(payload)
            if len(leaf_batch) >= leaf_batch_size:
                _eval_leaf_batch(leaf_batch, root, model, device, in_channels, value_dim, value_scale, min_max_stats)
                leaf_batch = []

    if leaf_batch:
        _eval_leaf_batch(leaf_batch, root, model, device, in_channels, value_dim, value_scale, min_max_stats)

    if not root.expanded:
        player = _player_from_name(name)
        priors, _ = policy_value(model, device, env, player, obs, in_channels, value_dim, value_scale)
        root.player = player
        root.expand(obs["action_mask"], priors)

    counts = root.nsa.astype(np.float32)
    mask = obs["action_mask"].astype(np.float32)
    counts = counts * mask
    prior = root.prior
    if prior is not None:
        prior = prior.astype(np.float32) * mask
        prior_sum = prior.sum()
        if prior_sum > 1e-8:
            prior = prior / prior_sum
    valid = root.valid_actions
    if valid is None or valid.size == 0:
        if prior is not None and prior.sum() > 1e-8:
            pi = prior
            return int(np.argmax(prior)), pi
        pi = mask
        if pi.sum() > 0:
            pi = pi / pi.sum()
        else:
            pi = np.ones_like(counts) / len(counts)
        return int(np.argmax(counts)), pi
    if temperature <= 1e-6:
        if counts.sum() <= 1e-8:
            if prior is not None and prior.sum() > 1e-8:
                return int(valid[int(np.argmax(prior[valid]))]), prior
            return int(valid[int(np.argmax(mask[valid]))]), mask / max(mask.sum(), 1.0)
        pi = counts / max(counts.sum(), 1.0)
        return int(valid[int(np.argmax(counts[valid]))]), pi
    probs = counts.copy()
    probs[probs < 0] = 0
    probs = probs ** (1.0 / temperature)
    if probs.sum() <= 1e-8:
        if prior is not None and prior.sum() > 1e-8:
            pi = prior
            return int(valid[int(np.argmax(prior[valid]))]), pi
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
    top_k,
    min_actions,
    policy_mass,
    leaf_batch_size,
    out_dir,
    save_every,
    log_interval,
    progress_path=None,
    progress_interval_sec=30,
    summary_path=None,
    wandb_run=None,
    wandb_log_interval=10,
    mcts_mode="all",
    mcts_player=-1,
    mcts_player_rotate=False,
    mcts_players=None,
    start_wall_limit=0,
    value_scale=1.0,
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
    mcts_win_count = 0
    non_mcts_win_count = 0
    mcts_episode_count = 0
    draw_count = 0
    invalid_count = 0

    in_channels = model.pre_conv[0].weight.shape[1]
    value_dim = model.value_head[-1].out_features
    rng = random.Random(seed)
    mcts_mode = str(mcts_mode).strip().lower()
    mcts_player = int(mcts_player)
    mcts_player_rotate = bool(mcts_player_rotate)
    start_wall_limit = int(start_wall_limit)
    if mcts_mode not in ("all", "single", "subset"):
        raise ValueError("mcts_mode must be all, single, or subset, got %s" % mcts_mode)
    if mcts_player >= 4 or mcts_player < -1:
        raise ValueError("mcts_player must be -1 or in [0, 3], got %s" % mcts_player)
    if mcts_mode == "single":
        if mcts_player >= 0:
            base_player = mcts_player
        else:
            base_player = (seed or 0) % 4
    if mcts_mode == "subset":
        base_players = _normalize_mcts_players(_parse_mcts_players(mcts_players))
        if not base_players:
            raise ValueError("mcts_mode=subset requires mcts_players")
        base_players = list(base_players)
        rotate_offset = (seed or 0) % 4
    samples = []
    file_idx = 0
    total_samples = 0
    start_time = time.time()
    last_progress_write = start_time

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if mcts_mode == "subset":
        players_repr = ",".join(str(p) for p in base_players)
    elif mcts_mode == "single":
        players_repr = str(base_player)
    else:
        players_repr = "all"
    print(
        "self-play config: episodes=%d sims=%d c_puct=%.3f determinize=%s temp=%.3f top_k=%s min_actions=%d "
        "policy_mass=%.2f leaf_batch=%d reward_scale=%.2f value_scale=%.2f out_dir=%s mcts_mode=%s mcts_players=%s start_wall_limit=%d"
        % (
            episodes,
            simulations,
            c_puct,
            str(determinize),
            temperature,
            str(top_k),
            min_actions,
            policy_mass,
            leaf_batch_size,
            reward_scale,
            value_scale,
            out_dir or "None",
            mcts_mode,
            players_repr,
            start_wall_limit,
        )
    )
    pbar = None
    if tqdm is not None:
        pbar = tqdm(range(episodes), unit="ep", dynamic_ncols=True)
        episode_iter = pbar
    else:
        episode_iter = range(episodes)

    for episode_idx in episode_iter:
        if mcts_mode == "all":
            mcts_player_ids = [0, 1, 2, 3]
            rotation_offset = 0
        elif mcts_mode == "single":
            if mcts_player_rotate:
                mcts_player_ids = [(base_player + episode_idx) % 4]
                rotation_offset = episode_idx % 4
            else:
                mcts_player_ids = [base_player]
                rotation_offset = 0
        else:
            if mcts_player_rotate:
                offset = (rotate_offset + episode_idx) % 4
                mcts_player_ids = [(p + offset) % 4 for p in base_players]
                rotation_offset = offset
            else:
                mcts_player_ids = list(base_players)
                rotation_offset = 0
        obs = env.reset()
        done = False
        rewards = None
        episode_samples = []
        mcts_used_in_episode = False
        while not done:
            if start_wall_limit > 0:
                mcts_active = _remaining_wall_tiles(env) <= start_wall_limit
            else:
                mcts_active = True
            if len(obs) == 1:
                name = next(iter(obs.keys()))
                player = _player_from_name(name)
                use_mcts = mcts_active and (player in mcts_player_ids)
                if use_mcts:
                    mcts_used_in_episode = True
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
                        top_k=top_k,
                        min_actions=min_actions,
                        policy_mass=policy_mass,
                        leaf_batch_size=leaf_batch_size,
                        mcts_player_ids=mcts_player_ids,
                        value_scale=value_scale,
                    )
                    episode_samples.append(
                        {
                            "oracle_obs": build_oracle_obs(env, player, obs[name]["observation"], in_channels),
                            "action_mask": obs[name]["action_mask"],
                            "pi": pi,
                            "player": player,
                        }
                    )
                else:
                    action = policy_action_single(model, device, env, player, obs[name], in_channels)
                action_dict = {n: 0 for n in agent_names}
                action_dict[name] = int(action)
            else:
                action_dict = {n: 0 for n in agent_names}
                action_dict.update(policy_actions(model, device, env, obs, in_channels))
            obs, rewards, done, err = _safe_env_step(env, action_dict)
            if err:
                invalid_count += 1
                rewards = {name: -30 for name in agent_names}
                break

        for name in agent_names:
            total_scores[name] += rewards.get(name, 0)

        reward_vals = [rewards.get(name, 0) for name in agent_names]
        if any(r == -30 for r in reward_vals):
            invalid_count += 1
            continue
        is_draw = all(r == 0 for r in reward_vals)
        if is_draw:
            draw_count += 1
        else:
            max_reward = max(reward_vals)
            if reward_vals.count(max_reward) == 1:
                winner = agent_names[reward_vals.index(max_reward)]
                orig_winner_player = _player_from_name(winner)
                winner_player = orig_winner_player
                if rotation_offset:
                    winner_player = (winner_player - rotation_offset) % 4
                winner_key = "player_%d" % (winner_player + 1)
                win_counts[winner_key] += 1
                if mcts_used_in_episode:
                    mcts_episode_count += 1
                    if orig_winner_player in mcts_player_ids:
                        mcts_win_count += 1
                    else:
                        non_mcts_win_count += 1

        if episode_samples:
            reward_vec = reward_to_vec(env, rewards, reward_scale)
            if is_draw and DRAW_PENALTY:
                reward_vec = reward_vec + DRAW_PENALTY
            for s in episode_samples:
                s["reward_vec"] = reward_vec
            if out_dir:
                samples.extend(episode_samples)
                total_samples += len(episode_samples)
                if len(samples) >= save_every:
                    _save_batch(out_dir, file_idx, samples)
                    file_idx += 1
                    samples = []

        now = time.time()
        if progress_path and (now - last_progress_write) >= progress_interval_sec:
            try:
                os.makedirs(os.path.dirname(progress_path), exist_ok=True)
                with open(progress_path, "w", encoding="utf-8") as f:
                    f.write(str(episode_idx + 1))
            except Exception:
                pass
            last_progress_write = now

        if log_interval and ((episode_idx + 1) % log_interval == 0):
            elapsed = time.time() - start_time
            avg_ep_time = elapsed / max(episode_idx + 1, 1)
            denom = max(mcts_episode_count, 1)
            win_rates = {
                "mcts": mcts_win_count / denom,
                "non_mcts": non_mcts_win_count / denom,
            }
            print(
                "episode %d/%d steps=%d samples=%d mcts_ep=%d wins_mcts=%d wins_non_mcts=%d win_rates=%s draws=%d invalid=%d avg_ep=%.2fs"
                % (
                    episode_idx + 1,
                    episodes,
                    len(episode_samples),
                    total_samples,
                    mcts_episode_count,
                    mcts_win_count,
                    non_mcts_win_count,
                    win_rates,
                    draw_count,
                    invalid_count,
                    avg_ep_time,
                )
            )

        if wandb_run and ((episode_idx + 1) % wandb_log_interval == 0):
            win_rates = {name: win_counts[name] / max(episode_idx + 1, 1) for name in agent_names}
            denom = max(mcts_episode_count, 1)
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
                    "mcts/episodes": mcts_episode_count,
                    "win_rate/mcts": mcts_win_count / denom,
                    "win_rate/non_mcts": non_mcts_win_count / denom,
                }
            )

        if pbar is not None:
            denom = max(mcts_episode_count, 1)
            mcts_rate = mcts_win_count / denom
            non_mcts_rate = non_mcts_win_count / denom
            win_rate_str = "%.2f/%.2f" % (mcts_rate, non_mcts_rate)
            pbar.set_postfix(samples=total_samples, draws=draw_count, invalid=invalid_count, win=win_rate_str, mcts_ep=mcts_episode_count)

    win_rates = {name: win_counts[name] / episodes for name in agent_names}
    if out_dir and samples:
        _save_batch(out_dir, file_idx, samples)
    if progress_path:
        try:
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, "w", encoding="utf-8") as f:
                f.write(str(episodes))
        except Exception:
            pass
    if summary_path:
        try:
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "episodes": episodes,
                        "mcts_episodes": mcts_episode_count,
                        "mcts_wins": mcts_win_count,
                        "non_mcts_wins": non_mcts_win_count,
                        "draws": draw_count,
                        "invalid": invalid_count,
                    },
                    f,
                )
        except Exception:
            pass
    if pbar is not None:
        pbar.close()
    return total_scores, win_rates, draw_count, invalid_count


def main():
    parser = argparse.ArgumentParser(description="Minimal MCTS self-play for oracle teacher (clean)")
    parser.add_argument("--model", required=True, help="Teacher checkpoint path")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dim fallback")
    parser.add_argument("--duplicate", action="store_true", help="Use duplicated wall env")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--simulations", type=int, default=64, help="MCTS sims per decision")
    parser.add_argument("--c_puct", type=float, default=1.5, help="PUCT constant")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Divide rewards by this value")
    parser.add_argument("--value_scale", type=float, default=1.0, help="Multiply value outputs by this value in MCTS")
    parser.add_argument("--determinize", action="store_true", help="Shuffle wall per simulation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Action temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-K actions to expand per node (0=disable)")
    parser.add_argument("--min_actions", type=int, default=6, help="Minimum actions to keep after pruning")
    parser.add_argument("--policy_mass", type=float, default=0.95, help="Policy mass threshold for pruning")
    parser.add_argument("--leaf_batch_size", type=int, default=16, help="Leaf evaluation batch size")
    parser.add_argument("--out_dir", default="", help="Save self-play samples to this dir")
    parser.add_argument("--save_every", type=int, default=4096, help="Samples per npz")
    parser.add_argument("--log_interval", type=int, default=1, help="Print every N episodes")
    parser.add_argument("--progress_path", default="", help="Write completed episodes to this file")
    parser.add_argument("--progress_interval_sec", type=int, default=30, help="Progress write interval (sec)")
    parser.add_argument("--summary_path", default="", help="Write summary stats to this file")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="mahjong-az", help="W&B project name")
    parser.add_argument("--wandb_run_name", default="", help="W&B run name")
    parser.add_argument("--wandb_log_interval", type=int, default=10, help="Log every N episodes")
    parser.add_argument("--mcts_mode", default="all", help="MCTS mode: all, single, or subset")
    parser.add_argument("--mcts_player", type=int, default=-1, help="MCTS player id (0-3), -1 for all players")
    parser.add_argument("--mcts_player_rotate", action="store_true", help="Rotate MCTS player each episode")
    parser.add_argument("--mcts_players", default="", help="Comma list of MCTS players for subset mode")
    parser.add_argument("--start_wall_limit", type=int, default=0, help="Use MCTS only when remaining tiles <= this")
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
                "value_scale": args.value_scale,
                "determinize": args.determinize,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "min_actions": args.min_actions,
                "policy_mass": args.policy_mass,
                "leaf_batch_size": args.leaf_batch_size,
                "mcts_mode": args.mcts_mode,
                "mcts_player": args.mcts_player,
                "mcts_player_rotate": args.mcts_player_rotate,
                "mcts_players": args.mcts_players,
                "start_wall_limit": args.start_wall_limit,
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
        value_scale=args.value_scale,
        determinize=args.determinize,
        temperature=args.temperature,
        top_k=args.top_k,
        min_actions=args.min_actions,
        policy_mass=args.policy_mass,
        leaf_batch_size=args.leaf_batch_size,
        out_dir=args.out_dir.strip() or None,
        save_every=args.save_every,
        log_interval=args.log_interval,
        progress_path=args.progress_path.strip() or None,
        progress_interval_sec=args.progress_interval_sec,
        summary_path=args.summary_path.strip() or None,
        wandb_run=wandb_run,
        wandb_log_interval=args.wandb_log_interval,
        mcts_mode=args.mcts_mode,
        mcts_player=args.mcts_player,
        mcts_player_rotate=args.mcts_player_rotate,
        mcts_players=args.mcts_players,
        start_wall_limit=args.start_wall_limit,
    )

    print("episodes:", args.episodes)
    print("draws:", draw_count, "invalid:", invalid_count)
    print("total_scores:", total_scores)
    print("win_rates:", win_rates)
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
