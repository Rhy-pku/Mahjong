#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a Mahjong model via self-play and report simple quality metrics.
"""

import argparse
import random
from collections import defaultdict

import numpy as np
import torch

from env import MahjongGBEnv
from feature_10m import FeatureAgent10M as FeatureAgent
from model_pretrain import PretrainModel


def load_state_dict_compat(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


def select_action(model, state, device):
    obs = torch.tensor(state["observation"], dtype=torch.float32, device=device).unsqueeze(0)
    vec = torch.tensor(state["vec"], dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.tensor(state["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs, vec, mask)
    return int(torch.argmax(logits, dim=1).item())


def evaluate(model, episodes, device, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = MahjongGBEnv(config={"agent_clz": FeatureAgent})
    agent_names = env.agent_names

    total_rewards = defaultdict(float)
    win_counts = defaultdict(int)
    steps_sum = 0
    hu_count = 0
    draw_count = 0
    invalid_count = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        rewards = None
        while not done:
            actions = {}
            for agent_name, state in obs.items():
                actions[agent_name] = select_action(model, state, device)
            obs, rewards, done = env.step(actions)
            steps += 1

        steps_sum += steps
        for name in agent_names:
            total_rewards[name] += rewards[name]

        reward_vals = [rewards[name] for name in agent_names]
        if any(r == -30 for r in reward_vals):
            invalid_count += 1
            continue
        if all(r == 0 for r in reward_vals):
            draw_count += 1
            continue

        hu_count += 1
        max_reward = max(reward_vals)
        if reward_vals.count(max_reward) == 1:
            winner = agent_names[reward_vals.index(max_reward)]
            win_counts[winner] += 1

    avg_rewards = {name: total_rewards[name] / episodes for name in agent_names}
    win_rates = {name: win_counts[name] / max(hu_count, 1) for name in agent_names}

    return {
        "episodes": episodes,
        "avg_steps": steps_sum / episodes,
        "hu_rate": hu_count / episodes,
        "draw_rate": draw_count / episodes,
        "invalid_rate": invalid_count / episodes,
        "avg_reward": avg_rewards,
        "win_rate": win_rates,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mahjong model via self-play")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Model hidden dim")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PretrainModel(hidden_dim=args.hidden_dim, use_vec=True).to(device)
    load_state_dict_compat(model, args.model)
    model.eval()

    metrics = evaluate(model, args.episodes, device, seed=args.seed)
    print("episodes:", metrics["episodes"])
    print("avg_steps:", round(metrics["avg_steps"], 2))
    print("hu_rate:", round(metrics["hu_rate"], 4))
    print("draw_rate:", round(metrics["draw_rate"], 4))
    print("invalid_rate:", round(metrics["invalid_rate"], 4))
    print("avg_reward:", {k: round(v, 3) for k, v in metrics["avg_reward"].items()})
    print("win_rate:", {k: round(v, 3) for k, v in metrics["win_rate"].items()})


if __name__ == "__main__":
    main()
