#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成纯模仿学习（Behavior Cloning）训练数据。

数据格式：
- obs: (N, 60, 4, 9) int8
- vec: (N, 117) float16
- mask: (N, 235) int8
- act: (N,) int16
"""

import argparse
import os
import sys
from typing import List

import numpy as np

# 添加父目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from feature_10m import FeatureAgent10M as FeatureAgent


def process_match(lines: List[str]) -> List[dict]:
    """处理单局游戏，返回该局所有样本（仅 policy 数据）"""
    agents = [FeatureAgent(i) for i in range(4)]
    player_samples: List[List[dict]] = [[] for _ in range(4)]
    cur_tile = None

    for line in lines:
        t = line.split()
        if not t:
            continue

        if t[0] == "Wind":
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == "Player":
            p = int(t[1])

            if t[2] == "Deal":
                agents[p].request2obs(" ".join(t[2:]))

            elif t[2] == "Draw":
                obs_dict = agents[p].request2obs(" ".join(t[2:]))
                if obs_dict is not None:
                    sample = {
                        "obs": obs_dict["observation"].copy(),
                        "vec": obs_dict["vec"].copy(),
                        "mask": obs_dict["action_mask"].copy(),
                        "act": 0,
                    }
                    player_samples[p].append(sample)
                for i in range(4):
                    if i != p:
                        agents[i].request2obs(" ".join(t[:3]))

            elif t[2] == "Play":
                cur_tile = t[3]
                if player_samples[p]:
                    act = agents[p].response2action(" ".join(t[2:]))
                    player_samples[p][-1]["act"] = act
                agents[p].request2obs(line)
                for i in range(4):
                    if i != p:
                        obs_dict = agents[i].request2obs(line)
                        if obs_dict is not None:
                            sample = {
                                "obs": obs_dict["observation"].copy(),
                                "vec": obs_dict["vec"].copy(),
                                "mask": obs_dict["action_mask"].copy(),
                                "act": 0,
                            }
                            player_samples[i].append(sample)

            elif t[2] == "Chi":
                if player_samples[p]:
                    act = agents[p].response2action(f"Chi {cur_tile} {t[3]}")
                    player_samples[p][-1]["act"] = act
                for i in range(4):
                    if i == p:
                        obs_dict = agents[p].request2obs(f"Player {p} Chi {t[3]}")
                        if obs_dict is not None:
                            sample = {
                                "obs": obs_dict["observation"].copy(),
                                "vec": obs_dict["vec"].copy(),
                                "mask": obs_dict["action_mask"].copy(),
                                "act": 0,
                            }
                            player_samples[p].append(sample)
                    else:
                        agents[i].request2obs(f"Player {p} Chi {t[3]}")

            elif t[2] == "Peng":
                if player_samples[p]:
                    act = agents[p].response2action(f"Peng {t[3]}")
                    player_samples[p][-1]["act"] = act
                for i in range(4):
                    if i == p:
                        obs_dict = agents[p].request2obs(f"Player {p} Peng {t[3]}")
                        if obs_dict is not None:
                            sample = {
                                "obs": obs_dict["observation"].copy(),
                                "vec": obs_dict["vec"].copy(),
                                "mask": obs_dict["action_mask"].copy(),
                                "act": 0,
                            }
                            player_samples[p].append(sample)
                    else:
                        agents[i].request2obs(f"Player {p} Peng {t[3]}")

            elif t[2] == "Gang":
                if player_samples[p]:
                    act = agents[p].response2action(f"Gang {t[3]}")
                    player_samples[p][-1]["act"] = act
                for i in range(4):
                    agents[i].request2obs(f"Player {p} Gang {t[3]}")

            elif t[2] == "AnGang":
                if player_samples[p]:
                    act = agents[p].response2action(f"AnGang {t[3]}")
                    player_samples[p][-1]["act"] = act
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(f"Player {p} AnGang {t[3]}")
                    else:
                        agents[i].request2obs(f"Player {p} AnGang")

            elif t[2] == "BuGang":
                if player_samples[p]:
                    act = agents[p].response2action(f"BuGang {t[3]}")
                    player_samples[p][-1]["act"] = act
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(f"Player {p} BuGang {t[3]}")
                    else:
                        obs_dict = agents[i].request2obs(f"Player {p} BuGang {t[3]}")
                        if obs_dict is not None:
                            sample = {
                                "obs": obs_dict["observation"].copy(),
                                "vec": obs_dict["vec"].copy(),
                                "mask": obs_dict["action_mask"].copy(),
                                "act": 0,
                            }
                            player_samples[i].append(sample)

            elif t[2] == "Hu":
                if player_samples[p]:
                    act = agents[p].response2action("Hu")
                    player_samples[p][-1]["act"] = act

            if t[2] in ["Peng", "Gang", "Hu"]:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p2 = int(t[k + 1])
                        if t[k + 2] == "Chi" and player_samples[p2]:
                            act = agents[p2].response2action(f"Chi {cur_tile} {t[k + 3]}")
                            player_samples[p2][-1]["act"] = act
                        elif t[k + 2] == "Peng" and player_samples[p2]:
                            act = agents[p2].response2action(f"Peng {t[k + 3]}")
                            player_samples[p2][-1]["act"] = act
                        elif t[k + 2] == "Gang" and player_samples[p2]:
                            act = agents[p2].response2action(f"Gang {t[k + 3]}")
                            player_samples[p2][-1]["act"] = act
                        elif t[k + 2] == "Hu" and player_samples[p2]:
                            act = agents[p2].response2action("Hu")
                            player_samples[p2][-1]["act"] = act
                    else:
                        break

    all_samples = []
    for p in range(4):
        all_samples.extend(player_samples[p])
    return all_samples


def _save_batch(output_dir: str, file_idx: int, obs, vec, mask, act):
    np.savez(
        os.path.join(output_dir, f"{file_idx}.npz"),
        obs=np.stack(obs).astype(np.int8),
        vec=np.stack(vec).astype(np.float16),
        mask=np.stack(mask).astype(np.int8),
        act=np.array(act, dtype=np.int16),
    )


def process_file(
    file_path: str,
    output_dir: str,
    samples_per_file: int = 50000,
    max_matches: int = None,
    verbose: bool = True,
):
    """处理整个文件（单进程，纯 BC 数据）"""
    os.makedirs(output_dir, exist_ok=True)

    all_obs, all_vec, all_mask, all_act = [], [], [], []
    file_idx = 0
    match_count = 0
    sample_count = 0

    with open(file_path, encoding="utf-8") as f:
        match_lines = []
        in_match = False

        for line in f:
            t = line.split()
            if not t:
                continue

            if t[0] == "Match":
                if in_match and match_lines:
                    samples = process_match(match_lines)
                    for s in samples:
                        all_obs.append(s["obs"])
                        all_vec.append(s["vec"])
                        all_mask.append(s["mask"])
                        all_act.append(s["act"])
                    sample_count += len(samples)
                    if len(all_obs) >= samples_per_file:
                        _save_batch(output_dir, file_idx, all_obs, all_vec, all_mask, all_act)
                        if verbose:
                            print(f"  Saved {output_dir}/{file_idx}.npz ({len(all_obs)} samples)")
                        all_obs, all_vec, all_mask, all_act = [], [], [], []
                        file_idx += 1

                match_lines = [line]
                in_match = True
                match_count += 1

                if max_matches and match_count > max_matches:
                    break
            else:
                match_lines.append(line)

        if in_match and match_lines and (not max_matches or match_count <= max_matches):
            samples = process_match(match_lines)
            for s in samples:
                all_obs.append(s["obs"])
                all_vec.append(s["vec"])
                all_mask.append(s["mask"])
                all_act.append(s["act"])
            sample_count += len(samples)

    if all_obs:
        _save_batch(output_dir, file_idx, all_obs, all_vec, all_mask, all_act)
        if verbose:
            print(f"  Saved {output_dir}/{file_idx}.npz ({len(all_obs)} samples)")

    if verbose:
        print(f"\nDone! Total: {match_count} matches, {sample_count} samples, {file_idx + 1} files")

    return {"matches": match_count, "samples": sample_count, "files": file_idx + 1}


def main():
    parser = argparse.ArgumentParser(description="Generate BC-only training data")
    parser.add_argument("--input", type=str, required=True, help="Input data.txt file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--samples_per_file", type=int, default=50000, help="Samples per output file")
    parser.add_argument("--max_matches", type=int, default=None, help="Max matches to process")
    args = parser.parse_args()

    process_file(
        args.input,
        args.output,
        samples_per_file=args.samples_per_file,
        max_matches=args.max_matches,
    )


if __name__ == "__main__":
    main()
