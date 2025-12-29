#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a single 1v3 game with MCTS for one player and log detailed search info.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

import mcts_self_play as mcts
from env import MahjongGBEnv
from feature_10m import FeatureAgent10M as FeatureAgent
from model_pretrain import PretrainModel


def _remaining_wall_tiles(env):
    wall = getattr(env, "tileWall", None)
    if not wall:
        return 0
    if isinstance(wall, list) and wall and isinstance(wall[0], list):
        return sum(len(sub) for sub in wall)
    return len(wall)


def _serialize_array(arr, top_k=0):
    arr = np.asarray(arr)
    if top_k and top_k > 0:
        order = np.argsort(-arr)
        idx = order[:top_k]
        return {
            "top_k": int(top_k),
            "indices": idx.tolist(),
            "values": arr[idx].tolist(),
        }
    return arr.tolist()


def _log_event(fp, payload):
    fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
    fp.flush()


def _select_with_stats(node, player, c_puct, top_k, min_actions, policy_mass, log_topk, log_compact, depth):
    n = max(node.n_visits, 1)
    valid = mcts._candidate_actions(node.prior, node.valid_actions, node.n_visits, top_k, min_actions, policy_mass)
    if valid is None or len(valid) == 0:
        valid = np.arange(node.action_size, dtype=np.int32)
    nsa = node.nsa[valid].astype(np.float32)
    wsa = node.wsa[valid, player]
    q_vals = wsa / np.maximum(nsa, 1.0)
    u_vals = c_puct * node.prior[valid] * (np.sqrt(n) / (1.0 + nsa))
    scores = q_vals + u_vals
    action = int(valid[int(np.argmax(scores))])
    stats = {
        "node_id": id(node),
        "player": int(player),
        "depth": int(depth),
        "n_visits": int(node.n_visits),
        "valid_actions": valid.tolist(),
        "prior_valid": node.prior[valid].astype(np.float32).tolist(),
        "nsa_valid": nsa.tolist(),
        "q_vals": q_vals.tolist(),
        "u_vals": u_vals.tolist(),
        "scores": scores.tolist(),
        "chosen_action": action,
    }
    if not log_compact:
        stats["prior"] = _serialize_array(node.prior, log_topk)
        stats["nsa"] = _serialize_array(node.nsa, log_topk)
        stats["wsa_player"] = _serialize_array(node.wsa[:, player], log_topk)
    return action, stats


def _simulate_to_leaf_debug(
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
    sim_idx,
    log_topk,
    log_compact,
):
    path = []
    node = root
    steps = []
    depth = 0
    while True:
        if env.done:
            value_vec = mcts.reward_to_vec(env, None, reward_scale)
            return "terminal", (value_vec, path), {"sim": sim_idx, "steps": steps}
        if len(obs_dict) != 1:
            action_dict = {name: 0 for name in env.agent_names}
            action_dict.update(mcts.policy_actions(model, device, env, obs_dict, in_channels))
            obs_dict, rewards, done, err = mcts._safe_env_step(env, action_dict)
            steps.append(
                {
                    "type": "multi_action",
                    "actions": {k: int(v) for k, v in action_dict.items()},
                    "done": bool(done),
                    "err": bool(err),
                }
            )
            if done:
                if err:
                    value_vec = np.zeros(4, dtype=np.float32)
                else:
                    value_vec = mcts.reward_to_vec(env, rewards, reward_scale)
                return "terminal", (value_vec, path), {"sim": sim_idx, "steps": steps}
            continue

        name, obs = next(iter(obs_dict.items()))
        player = mcts._player_from_name(name)
        if not node.expanded:
            return "leaf", (node, path, player, obs, env), {"sim": sim_idx, "steps": steps}

        use_mcts = mcts_player_ids is None or player in mcts_player_ids
        if use_mcts:
            action, stats = _select_with_stats(
                node,
                player,
                c_puct,
                top_k,
                min_actions,
                policy_mass,
                log_topk,
                log_compact,
                depth,
            )
            stats["policy"] = "mcts_select"
        else:
            action = int(np.argmax(node.prior))
            if log_compact:
                valid = node.valid_actions
                if valid is None or len(valid) == 0:
                    valid = np.arange(node.action_size, dtype=np.int32)
                stats = {
                    "node_id": id(node),
                    "player": int(player),
                    "depth": int(depth),
                    "policy": "prior_argmax",
                    "valid_actions": valid.tolist(),
                    "prior_valid": node.prior[valid].astype(np.float32).tolist(),
                    "chosen_action": action,
                }
            else:
                stats = {
                    "node_id": id(node),
                    "player": int(player),
                    "depth": int(depth),
                    "policy": "prior_argmax",
                    "prior": _serialize_array(node.prior, log_topk),
                    "nsa": _serialize_array(node.nsa, log_topk),
                    "wsa_player": _serialize_array(node.wsa[:, player], log_topk),
                    "chosen_action": action,
                }
        try:
            response = env.agents[player].action2response(action)
        except Exception:
            response = None
        stats["action_response"] = response
        steps.append(stats)

        path.append((node, action))
        child = node.children.get(action)
        if child is None:
            child = mcts.MCTSNode(node.action_size, node.value_dim)
            node.children[action] = child
        node = child
        depth += 1
        action_dict = {n: 0 for n in env.agent_names}
        action_dict[name] = action
        obs_dict, rewards, done, err = mcts._safe_env_step(env, action_dict)
        if done:
            if err:
                value_vec = np.zeros(4, dtype=np.float32)
            else:
                value_vec = mcts.reward_to_vec(env, rewards, reward_scale)
            return "terminal", (value_vec, path), {"sim": sim_idx, "steps": steps}


def _eval_leaf_batch_debug(
    leaf_batch,
    leaf_meta,
    root,
    model,
    device,
    in_channels,
    value_dim,
    value_scale,
    log_topk,
    log_obs,
    log_compact,
):
    obs_list = []
    mask_list = []
    players = []
    nodes = []
    paths = []
    for node, path, player, obs, env in leaf_batch:
        oracle_obs = mcts.build_oracle_obs(env, player, obs["observation"], in_channels)
        obs_list.append(oracle_obs)
        mask_list.append(obs["action_mask"])
        players.append(player)
        nodes.append(node)
        paths.append(path)

    obs_t = mcts._to_device(np.stack(obs_list), device)
    mask_t = mcts._to_device(np.stack(mask_list), device)
    with torch.inference_mode():
        logits, values = model(obs_t)
    masked_logits = mcts._apply_action_mask(logits, mask_t)
    probs = torch.softmax(masked_logits, dim=1).cpu().numpy()
    values_np = values.cpu().numpy()

    logs = []
    for i in range(len(leaf_batch)):
        priors = probs[i]
        nodes[i].player = players[i]
        nodes[i].expand(mask_list[i], priors)
        if value_dim == 1:
            value_vec = np.zeros(4, dtype=np.float32)
            value_vec[players[i]] = float(values_np[i, 0]) * float(value_scale)
        else:
            value_vec = values_np[i].astype(np.float32) * float(value_scale)
        mcts._backprop(paths[i], root, value_vec)
        if log_compact:
            valid = np.flatnonzero(mask_list[i] > 0)
            compact_topk = log_topk if log_topk and log_topk > 0 else 8
            entry = {
                "type": "leaf_eval",
                "sim": leaf_meta[i]["sim"],
                "player": int(players[i]),
                "steps": leaf_meta[i].get("steps", []),
                "valid_actions": valid.tolist(),
                "priors_topk": _serialize_array(priors, compact_topk),
                "value_vec": _serialize_array(value_vec, log_topk),
            }
        else:
            entry = {
                "type": "leaf_eval",
                "sim": leaf_meta[i]["sim"],
                "player": int(players[i]),
                "steps": leaf_meta[i].get("steps", []),
                "action_mask": _serialize_array(mask_list[i], log_topk),
                "priors": _serialize_array(priors, log_topk),
                "value_vec": _serialize_array(value_vec, log_topk),
            }
        if log_obs:
            entry["oracle_obs"] = _serialize_array(obs_list[i], 0)
        logs.append(entry)
    return logs


def mcts_action_debug(
    env,
    obs_dict,
    model,
    device,
    in_channels,
    value_dim,
    value_scale,
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
    mcts_player_ids,
    step=0,
    log_topk=0,
    log_obs=False,
    log_compact=True,
    tree_path=None,
    tree_depth=3,
    tree_max_children=8,
    tree_min_visits=1,
    log_fp=None,
):
    name, obs = next(iter(obs_dict.items()))
    player = mcts._player_from_name(name)
    action_size = obs["action_mask"].shape[0]
    hu_action = mcts._get_hu_action(obs["action_mask"])
    if hu_action is not None:
        pi = np.zeros(action_size, dtype=np.float32)
        pi[hu_action] = 1.0
        _log_event(
            log_fp,
            {
                "type": "decision_end",
                "reason": "hu_action",
                "step": int(step),
                "player": int(player),
                "name": name,
                "use_mcts": True,
                "action": int(hu_action),
                "pi": _serialize_array(pi, log_topk),
            },
        )
        return int(hu_action), pi

    root = mcts.MCTSNode(action_size, value_dim)
    leaf_batch = []
    leaf_meta = []
    leaf_batch_size = max(1, int(leaf_batch_size))
    if mcts_player_ids is not None:
        mcts_player_ids = set(mcts_player_ids)

    for sim in range(simulations):
        env_copy = mcts._clone_env(env)
        if determinize:
            mcts._shuffle_wall(env_copy, rng)
        obs_copy = env_copy._obs()
        kind, payload, sim_meta = _simulate_to_leaf_debug(
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
            sim,
            log_topk,
            log_compact,
        )
        if kind == "terminal":
            value_vec, path = payload
            mcts._backprop(path, root, value_vec)
            sim_meta["type"] = "simulation_terminal"
            sim_meta["value_vec"] = _serialize_array(value_vec, log_topk)
            _log_event(log_fp, sim_meta)
        else:
            leaf_batch.append(payload)
            leaf_meta.append(sim_meta)
            if len(leaf_batch) >= leaf_batch_size:
                logs = _eval_leaf_batch_debug(
                    leaf_batch,
                    leaf_meta,
                    root,
                    model,
                    device,
                    in_channels,
                    value_dim,
                    value_scale,
                    log_topk,
                    log_obs,
                    log_compact,
                )
                for entry in logs:
                    _log_event(log_fp, entry)
                leaf_batch = []
                leaf_meta = []

    if leaf_batch:
        logs = _eval_leaf_batch_debug(
            leaf_batch,
            leaf_meta,
            root,
            model,
            device,
            in_channels,
            value_dim,
            value_scale,
            log_topk,
            log_obs,
            log_compact,
        )
        for entry in logs:
            _log_event(log_fp, entry)

    if not root.expanded:
        player = mcts._player_from_name(name)
        priors, _ = mcts.policy_value(model, device, env, player, obs, in_channels, value_dim, value_scale)
        root.player = player
        root.expand(obs["action_mask"], priors)
        _log_event(
            log_fp,
            {
                "type": "root_expand_fallback",
                "priors": _serialize_array(priors, log_topk),
            },
        )

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
        action = int(np.argmax(counts))
    elif temperature <= 1e-6:
        if counts.sum() <= 1e-8:
            action = int(valid[int(np.argmax(mask[valid]))])
            pi = mask / max(mask.sum(), 1.0)
        else:
            pi = counts / max(counts.sum(), 1.0)
            action = int(valid[int(np.argmax(counts[valid]))])
    else:
        probs = counts.copy()
        probs[probs < 0] = 0
        probs = probs ** (1.0 / temperature)
        if probs.sum() <= 1e-8:
            pi = mask
            if pi.sum() > 0:
                pi = pi / pi.sum()
            else:
                pi = np.ones_like(counts) / len(counts)
            action = int(valid[int(np.argmax(counts[valid]))])
        else:
            probs /= probs.sum()
            pi = probs
            action = int(np.random.choice(np.arange(len(probs)), p=probs))

    decision_log = {
        "type": "decision_end",
        "step": int(step),
        "player": int(player),
        "name": name,
        "use_mcts": True,
        "action": int(action),
        "pi": _serialize_array(pi, log_topk),
        "root_n_visits": int(root.n_visits),
    }
    if log_compact:
        if root.valid_actions is not None:
            valid = root.valid_actions
        else:
            valid = np.flatnonzero(obs["action_mask"] > 0)
        decision_log["root_valid"] = valid.tolist()
        if root.prior is not None:
            decision_log["root_prior_valid"] = root.prior[valid].astype(np.float32).tolist()
        decision_log["root_nsa_valid"] = root.nsa[valid].astype(np.int32).tolist()
        root_player = root.player if root.player is not None else player
        q_vals = root.wsa[valid, root_player] / np.maximum(root.nsa[valid], 1.0)
        decision_log["root_q_valid"] = q_vals.astype(np.float32).tolist()
    else:
        decision_log["root_prior"] = _serialize_array(root.prior, log_topk)
        decision_log["root_nsa"] = _serialize_array(root.nsa, log_topk)
        decision_log["root_wsa_player"] = _serialize_array(root.wsa[:, mcts._player_from_name(name)], log_topk)
        decision_log["root_valid"] = root.valid_actions.tolist() if root.valid_actions is not None else []
    _log_event(log_fp, decision_log)
    if tree_path:
        tree_out = mcts._format_debug_tree_path(tree_path, 0, step)
        mcts._dump_tree_dot(
            root,
            tree_out,
            max_depth=int(tree_depth),
            max_children=int(tree_max_children),
            min_visits=int(tree_min_visits),
        )
    return int(action), pi


def main():
    parser = argparse.ArgumentParser(description="Debug single MCTS 1v3 game with detailed logs")
    parser.add_argument("--model", required=True, help="Teacher checkpoint path")
    parser.add_argument("--log_path", required=True, help="Output log file path")
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
    parser.add_argument("--mcts_player", type=int, default=0, help="MCTS player id (0-3)")
    parser.add_argument("--start_wall_limit", type=int, default=0, help="Use MCTS only when remaining tiles <= this")
    parser.add_argument("--log_topk", type=int, default=0, help="Log only top-K entries for arrays (0=full)")
    parser.add_argument("--log_obs", action="store_true", help="Log oracle observations in detail")
    parser.add_argument("--log_full", action="store_true", help="Log full arrays in debug output")
    parser.add_argument("--tree_path", default="", help="Write MCTS tree DOT per decision (supports {step})")
    parser.add_argument("--tree_depth", type=int, default=3, help="Max depth for debug tree")
    parser.add_argument("--tree_max_children", type=int, default=8, help="Max children per node in debug tree")
    parser.add_argument("--tree_min_visits", type=int, default=1, help="Min visits to include edge in debug tree")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N env steps (0=disable)")
    args = parser.parse_args()
    args.log_compact = not args.log_full

    device = torch.device(args.device)
    state_dict = mcts.load_state_dict_compat(args.model)
    in_channels = mcts.infer_in_channels(state_dict)
    hidden_dim = mcts.infer_hidden_dim(state_dict, args.hidden_dim)
    value_dim = mcts.infer_value_dim(state_dict)
    model = PretrainModel(
        hidden_dim=hidden_dim,
        use_vec=False,
        in_channels=in_channels,
        value_dim=value_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(args.log_path, "w", encoding="utf-8") as log_fp:
        _log_event(
            log_fp,
            {
                "type": "config",
                "model": args.model,
                "device": args.device,
                "seed": args.seed,
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
                "mcts_player": args.mcts_player,
                "start_wall_limit": args.start_wall_limit,
                "log_topk": args.log_topk,
                "log_obs": args.log_obs,
                "log_compact": args.log_compact,
                "tree_path": args.tree_path,
                "tree_depth": args.tree_depth,
                "tree_max_children": args.tree_max_children,
                "tree_min_visits": args.tree_min_visits,
                "in_channels": in_channels,
                "value_dim": value_dim,
                "timestamp": time.time(),
            },
        )

        env = MahjongGBEnv(config={"agent_clz": FeatureAgent, "duplicate": args.duplicate})
        obs = env.reset()
        done = False
        step_idx = 0
        rng = random.Random(args.seed)

        _log_event(log_fp, {"type": "episode_start"})

        while not done:
            if args.max_steps and step_idx >= args.max_steps:
                _log_event(log_fp, {"type": "max_steps_reached", "step": step_idx})
                break
            if args.start_wall_limit > 0:
                mcts_active = _remaining_wall_tiles(env) <= args.start_wall_limit
            else:
                mcts_active = True

            if len(obs) == 1:
                name = next(iter(obs.keys()))
                player = mcts._player_from_name(name)
                use_mcts = mcts_active and (player == args.mcts_player)
                decision_start = {
                    "type": "decision_start",
                    "step": step_idx,
                    "player": int(player),
                    "name": name,
                    "use_mcts": bool(use_mcts),
                    "state": int(getattr(env, "state", -1)),
                    "cur_tile": getattr(env, "curTile", None),
                    "wall_remaining": _remaining_wall_tiles(env),
                    "hand": list(env.hands[player]),
                }
                if args.log_compact:
                    valid_actions = np.flatnonzero(obs[name]["action_mask"] > 0)
                    decision_start["valid_actions"] = valid_actions.tolist()
                    decision_start["valid_count"] = int(valid_actions.size)
                else:
                    decision_start["action_mask"] = _serialize_array(obs[name]["action_mask"], args.log_topk)
                _log_event(log_fp, decision_start)
                if args.log_obs:
                    oracle_obs = mcts.build_oracle_obs(env, player, obs[name]["observation"], in_channels)
                    _log_event(
                        log_fp,
                        {
                            "type": "oracle_obs",
                            "step": step_idx,
                            "player": int(player),
                            "oracle_obs": _serialize_array(oracle_obs, 0),
                        },
                    )
                if use_mcts:
                    action, pi = mcts_action_debug(
                        env=env,
                        obs_dict=obs,
                        model=model,
                        device=device,
                        in_channels=in_channels,
                        value_dim=value_dim,
                        value_scale=args.value_scale,
                        simulations=args.simulations,
                        c_puct=args.c_puct,
                        reward_scale=args.reward_scale,
                        determinize=args.determinize,
                        rng=rng,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        min_actions=args.min_actions,
                        policy_mass=args.policy_mass,
                        leaf_batch_size=args.leaf_batch_size,
                        mcts_player_ids=[args.mcts_player],
                        step=step_idx,
                        log_topk=args.log_topk,
                        log_obs=args.log_obs,
                        log_compact=args.log_compact,
                        tree_path=args.tree_path,
                        tree_depth=args.tree_depth,
                        tree_max_children=args.tree_max_children,
                        tree_min_visits=args.tree_min_visits,
                        log_fp=log_fp,
                    )
                else:
                    action = mcts.policy_action_single(model, device, env, player, obs[name], in_channels)
                    pi = None
                    _log_event(
                        log_fp,
                        {
                            "type": "decision_end",
                            "reason": "non_mcts_policy",
                            "step": step_idx,
                            "player": int(player),
                            "name": name,
                            "use_mcts": False,
                            "action": int(action),
                            "pi": None,
                        },
                    )
                try:
                    response = env.agents[player].action2response(action)
                except Exception:
                    response = None
                action_dict = {n: 0 for n in env.agent_names}
                action_dict[name] = int(action)
                obs, rewards, done, err = mcts._safe_env_step(env, action_dict)
                _log_event(
                    log_fp,
                    {
                        "type": "env_step",
                        "step": step_idx,
                        "action_dict": {k: int(v) for k, v in action_dict.items()},
                        "action_response": response,
                        "rewards": rewards,
                        "done": bool(done),
                        "err": bool(err),
                    },
                )
            else:
                action_dict = {n: 0 for n in env.agent_names}
                action_dict.update(mcts.policy_actions(model, device, env, obs, in_channels))
                obs, rewards, done, err = mcts._safe_env_step(env, action_dict)
                _log_event(
                    log_fp,
                    {
                        "type": "env_step_multi",
                        "step": step_idx,
                        "action_dict": {k: int(v) for k, v in action_dict.items()},
                        "rewards": rewards,
                        "done": bool(done),
                        "err": bool(err),
                    },
                )

            step_idx += 1

        _log_event(
            log_fp,
            {
                "type": "episode_end",
                "steps": step_idx,
                "done": bool(done),
                "reward": getattr(env, "reward", None),
            },
        )


if __name__ == "__main__":
    main()
