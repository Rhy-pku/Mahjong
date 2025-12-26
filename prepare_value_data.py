#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 Value Head 训练数据

从原始对局数据中提取特征和最终得分，用于预训练 Value Head。
基于 preprocess.py 的逻辑，但添加 Score 信息。

输入: data.txt (Botzone 对局记录)
输出: value_data/*.npz (obs, vec, score)

用法:
    # 测试
    python prepare_value_data.py --input /root/data/sample.txt --output /root/autodl-tmp/value_data_test
    
    # 完整数据
    python prepare_value_data.py --input /root/data/data.txt --output /root/autodl-tmp/value_data
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Tuple
import numpy as np
import math
import multiprocessing as mp

# 设置路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT_DIR)

from feature_10m import FeatureAgent10M as FeatureAgent


def compute_score_stats(file_path: str) -> Dict:
    """两遍处理：先统计得分均值/方差，用于 z-score"""
    count = 0
    mean = 0.0
    m2 = 0.0
    smin = math.inf
    smax = -math.inf
    
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if len(t) == 0:
                continue
            if t[0] == "Score":
                scores = [int(x) / 100.0 for x in t[1:5]]
                for s in scores:
                    count += 1
                    delta = s - mean
                    mean += delta / count
                    delta2 = s - mean
                    m2 += delta * delta2
                    smin = min(smin, s)
                    smax = max(smax, s)
    
    std = math.sqrt(m2 / (count - 1)) if count > 1 else 1.0
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": smin,
        "max": smax,
    }


def filterData(obs, player_ids):
    """过滤只剩下可行动作大于1的状态"""
    newobs = []
    newplayers = []
    for i, o in enumerate(obs):
        if o["action_mask"].sum() > 1:
            newobs.append(o)
            newplayers.append(player_ids[i])
    return newobs, newplayers


def _process_match_block(block_lines: List[str], score_stats: Dict, eps: float) -> Tuple[List, List, List, int]:
    """处理单个对局块，返回 obs/vec/score 列表和 match 计数"""
    agents = [FeatureAgent(i) for i in range(4)]
    obs = []
    player_ids = []
    all_obs = []
    all_vec = []
    all_score = []
    match_count = 0
    
    for line in block_lines:
        t = line.split()
        if len(t) == 0:
            continue
        if t[0] == "Match":
            match_count += 1
            agents = [FeatureAgent(i) for i in range(4)]
            obs = []
            player_ids = []
        elif t[0] == "Wind":
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == "Player":
            p = int(t[1])
            if t[2] == "Deal":
                agents[p].request2obs(" ".join(t[2:]))
            elif t[2] == "Draw":
                o = agents[p].request2obs(" ".join(t[2:]))
                obs.append(o)
                player_ids.append(p)
                for i in range(4):
                    if i != p:
                        agents[i].request2obs(" ".join(t[:3]))
            elif t[2] == "Play":
                curTile = t[3]
                agents[p].request2obs(line)
                for i in range(4):
                    if i != p:
                        o = agents[i].request2obs(line)
                        obs.append(o)
                        player_ids.append(i)
            elif t[2] == "Chi":
                for i in range(4):
                    if i == p:
                        o = agents[p].request2obs(f"Player {p} Chi {t[3]}")
                        obs.append(o)
                        player_ids.append(p)
                    else:
                        agents[i].request2obs(f"Player {p} Chi {t[3]}")
            elif t[2] == "Peng":
                for i in range(4):
                    if i == p:
                        o = agents[p].request2obs(f"Player {p} Peng {t[3]}")
                        obs.append(o)
                        player_ids.append(p)
                    else:
                        agents[i].request2obs(f"Player {p} Peng {t[3]}")
            elif t[2] == "Gang":
                for i in range(4):
                    agents[i].request2obs(f"Player {p} Gang {t[3]}")
            elif t[2] == "AnGang":
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(f"Player {p} AnGang {t[3]}")
                    else:
                        agents[i].request2obs(f"Player {p} AnGang")
            elif t[2] == "BuGang":
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(f"Player {p} BuGang {t[3]}")
                    else:
                        o = agents[i].request2obs(f"Player {p} BuGang {t[3]}")
                        obs.append(o)
                        player_ids.append(i)
            elif t[2] == "Hu":
                pass
        elif t[0] == "Score":
            scores_raw = np.array([int(x) for x in t[1:5]], dtype=np.float32) / 100.0
            obs, player_ids = filterData(obs, player_ids)
            normalized_scores = (scores_raw - score_stats["mean"]) / max(score_stats["std"], eps)
            for i, o in enumerate(obs):
                all_obs.append(o["observation"])
                all_vec.append(o["vec"])
                all_score.append(normalized_scores[player_ids[i]])
            obs = []
            player_ids = []
    return all_obs, all_vec, all_score, match_count


def process_data(file_path: str, output_dir: str, samples_per_file: int = 50000):
    """处理数据文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 第一次遍历：计算分数统计量（用于 z-score）
    score_stats = compute_score_stats(file_path)
    eps = 1e-8
    print("Score stats:")
    print(json.dumps(score_stats, indent=2, ensure_ascii=False))
    
    # 第二遍：并行处理每个 Match 块
    print(f"Processing {file_path} in parallel...")
    all_obs = []
    all_vec = []
    all_score = []
    file_idx = 0
    total_samples = 0
    match_count = 0
    
    def flush():
        nonlocal all_obs, all_vec, all_score, file_idx, total_samples
        if all_obs:
            save_file(output_dir, file_idx, all_obs, all_vec, all_score)
            total_samples += len(all_obs)
            file_idx += 1
            all_obs = []
            all_vec = []
            all_score = []
    
    # 构造按 Match 分块的列表
    blocks = []
    current_block = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Match") and current_block:
                blocks.append(current_block)
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            blocks.append(current_block)
    
    # 并行处理
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(_process_match_block, args=(blk, score_stats, eps)) for blk in blocks]
    pool.close()
    for r in results:
        o, v, s, mc = r.get()
        match_count += mc
        all_obs.extend(o)
        all_vec.extend(v)
        all_score.extend(s)
        if len(all_obs) >= samples_per_file:
            flush()
    flush()
    pool.join()
    
    print(f"\nDone!")
    print(f"  Total matches: {match_count}")
    print(f"  Total samples: {total_samples}")
    print(f"  Output files: {file_idx}")
    print(f"  Output dir: {output_dir}")
    
    # 保存统计信息
    stats = {
        "total_matches": match_count,
        "total_samples": total_samples,
        "num_files": file_idx,
        "score_stats": score_stats,
        "normalize": "zscore",
    }
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def save_file(output_dir: str, file_idx: int, obs_list: List, vec_list: List, score_list: List):
    """保存数据文件"""
    obs = np.array(obs_list, dtype=np.int8)
    vec = np.array(vec_list, dtype=np.float16)
    score = np.array(score_list, dtype=np.float32)
    
    output_path = os.path.join(output_dir, f"{file_idx}.npz")
    np.savez_compressed(output_path, obs=obs, vec=vec, score=score)
    print(f"  Saved {output_path}: {len(obs_list)} samples, obs={obs.shape}, score range=[{score.min():.2f}, {score.max():.2f}]")


def main():
    parser = argparse.ArgumentParser(description="Prepare Value Head training data")
    parser.add_argument("--input", type=str, required=True, help="输入数据文件")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--samples_per_file", type=int, default=100000, help="每个文件的样本数")
    
    args = parser.parse_args()
    
    process_data(args.input, args.output, args.samples_per_file)


if __name__ == "__main__":
    main()
