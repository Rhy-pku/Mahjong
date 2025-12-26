#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成同时训练 policy 和 value head 的完整数据。

数据格式：
- obs: (N, 60, 4, 9) int8 - 观察特征
- vec: (N, 117) float16 - 向量特征
- mask: (N, 235) int8 - 动作掩码
- act: (N,) int16 - 专家动作
- value_target: (N,) float32 - 折扣回报（用于 value head 训练）

Value target 计算：
- γ^(T-t) * normalized_score
- γ: 折扣因子（默认 0.99）
- T: 该局该玩家的最后一步
- t: 当前步数
- normalized_score: z-score 归一化后的局终得分
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# 添加父目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from feature_10m import FeatureAgent10M as FeatureAgent


def filter_data(samples: List[dict]) -> List[dict]:
    """过滤只保留可行动作大于1的状态"""
    return [s for s in samples if s["mask"].sum() > 1]


def process_match(lines: List[str], gamma: float, score_stats: Dict) -> List[dict]:
    """
    处理单局游戏，返回该局所有样本。
    
    Args:
        lines: 该局的所有日志行
        gamma: 折扣因子
        score_stats: {"mean": float, "std": float} 用于 z-score 归一化
    
    Returns:
        样本列表，每个样本包含 obs, vec, mask, act, value_target
    """
    agents = [FeatureAgent(i) for i in range(4)]
    
    # 收集每个玩家的样本（带 step_idx）
    player_samples = [[] for _ in range(4)]  # player_samples[p] = [(sample_dict, step_idx), ...]
    step_counter = [0, 0, 0, 0]  # 每个玩家的步数计数
    
    final_scores = None
    curTile = None
    
    for line in lines:
        t = line.split()
        if len(t) == 0:
            continue
            
        if t[0] == "Wind":
            for agent in agents:
                agent.request2obs(line)
                
        elif t[0] == "Player":
            p = int(t[1])
            
            if t[2] == "Deal":
                agents[p].request2obs(" ".join(t[2:]))
                
            elif t[2] == "Draw":
                # 玩家 p 摸牌，产生决策点
                obs_dict = agents[p].request2obs(" ".join(t[2:]))
                if obs_dict is not None:
                    sample = {
                        "obs": obs_dict["observation"].copy(),
                        "vec": obs_dict["vec"].copy(),
                        "mask": obs_dict["action_mask"].copy(),
                        "act": 0,  # 占位，等 Play 时更新
                    }
                    player_samples[p].append((sample, step_counter[p]))
                    step_counter[p] += 1
                # 其他玩家更新状态
                for i in range(4):
                    if i != p:
                        agents[i].request2obs(" ".join(t[:3]))
                        
            elif t[2] == "Play":
                curTile = t[3]
                # 更新玩家 p 的动作
                if player_samples[p]:
                    act = agents[p].response2action(" ".join(t[2:]))
                    player_samples[p][-1][0]["act"] = act
                # 更新状态
                agents[p].request2obs(line)
                # 其他玩家产生决策点
                for i in range(4):
                    if i != p:
                        obs_dict = agents[i].request2obs(line)
                        if obs_dict is not None:
                            sample = {
                                "obs": obs_dict["observation"].copy(),
                                "vec": obs_dict["vec"].copy(),
                                "mask": obs_dict["action_mask"].copy(),
                                "act": 0,  # Pass
                            }
                            player_samples[i].append((sample, step_counter[i]))
                            step_counter[i] += 1
                            
            elif t[2] == "Chi":
                # 更新玩家 p 的动作（替换之前的 Pass）
                if player_samples[p]:
                    act = agents[p].response2action(f"Chi {curTile} {t[3]}")
                    player_samples[p][-1][0]["act"] = act
                # 玩家 p 产生新决策点（吃完要出牌）
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
                            player_samples[p].append((sample, step_counter[p]))
                            step_counter[p] += 1
                    else:
                        agents[i].request2obs(f"Player {p} Chi {t[3]}")
                        
            elif t[2] == "Peng":
                # 更新玩家 p 的动作
                if player_samples[p]:
                    act = agents[p].response2action(f"Peng {t[3]}")
                    player_samples[p][-1][0]["act"] = act
                # 玩家 p 产生新决策点
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
                            player_samples[p].append((sample, step_counter[p]))
                            step_counter[p] += 1
                    else:
                        agents[i].request2obs(f"Player {p} Peng {t[3]}")
                        
            elif t[2] == "Gang":
                # 更新玩家 p 的动作
                if player_samples[p]:
                    act = agents[p].response2action(f"Gang {t[3]}")
                    player_samples[p][-1][0]["act"] = act
                for i in range(4):
                    agents[i].request2obs(f"Player {p} Gang {t[3]}")
                    
            elif t[2] == "AnGang":
                if player_samples[p]:
                    act = agents[p].response2action(f"AnGang {t[3]}")
                    player_samples[p][-1][0]["act"] = act
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(f"Player {p} AnGang {t[3]}")
                    else:
                        agents[i].request2obs(f"Player {p} AnGang")
                        
            elif t[2] == "BuGang":
                if player_samples[p]:
                    act = agents[p].response2action(f"BuGang {t[3]}")
                    player_samples[p][-1][0]["act"] = act
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
                            player_samples[i].append((sample, step_counter[i]))
                            step_counter[i] += 1
                            
            elif t[2] == "Hu":
                if player_samples[p]:
                    act = agents[p].response2action("Hu")
                    player_samples[p][-1][0]["act"] = act
                    
            # 处理同一行的其他动作（Peng/Gang/Hu 可能有多个玩家响应）
            if t[2] in ["Peng", "Gang", "Hu"]:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p2 = int(t[k + 1])
                        if t[k + 2] == "Chi" and player_samples[p2]:
                            act = agents[p2].response2action(f"Chi {curTile} {t[k + 3]}")
                            player_samples[p2][-1][0]["act"] = act
                        elif t[k + 2] == "Peng" and player_samples[p2]:
                            act = agents[p2].response2action(f"Peng {t[k + 3]}")
                            player_samples[p2][-1][0]["act"] = act
                        elif t[k + 2] == "Gang" and player_samples[p2]:
                            act = agents[p2].response2action(f"Gang {t[k + 3]}")
                            player_samples[p2][-1][0]["act"] = act
                        elif t[k + 2] == "Hu" and player_samples[p2]:
                            act = agents[p2].response2action("Hu")
                            player_samples[p2][-1][0]["act"] = act
                    else:
                        break
                        
        elif t[0] == "Score":
            # 解析最终得分
            final_scores = np.array([int(x) for x in t[1:5]], dtype=np.float32) / 100.0
    
    if final_scores is None:
        return []
    
    # 计算折扣回报并汇总所有样本
    all_samples = []
    eps = 1e-8
    
    for p in range(4):
        samples = player_samples[p]
        if not samples:
            continue
            
        # 该玩家的归一化得分
        norm_score = (final_scores[p] - score_stats["mean"]) / max(score_stats["std"], eps)
        
        # 该玩家的总步数
        T = len(samples)
        
        for sample, step_idx in samples:
            # 折扣回报: γ^(T-1-step_idx) * norm_score
            # step_idx 从 0 开始，最后一步是 T-1
            discount = gamma ** (T - 1 - step_idx)
            sample["value_target"] = discount * norm_score
            all_samples.append(sample)
    
    # 过滤只保留有效决策点
    all_samples = filter_data(all_samples)
    
    return all_samples


def compute_score_stats(file_path: str, max_matches: int = None) -> Dict:
    """两遍扫描：第一遍计算分数统计"""
    scores = []
    with open(file_path, encoding="utf-8") as f:
        match_count = 0
        for line in f:
            t = line.split()
            if len(t) == 0:
                continue
            if t[0] == "Score":
                for x in t[1:5]:
                    scores.append(int(x) / 100.0)
                match_count += 1
                if max_matches and match_count >= max_matches:
                    break
    
    scores = np.array(scores, dtype=np.float32)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "count": len(scores),
    }


def process_file(
    file_path: str,
    output_dir: str,
    gamma: float = 0.99,
    samples_per_file: int = 50000,
    max_matches: int = None,
    verbose: bool = True,
):
    """处理整个文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 第一遍：计算分数统计
    if verbose:
        print("Pass 1: Computing score statistics...")
    score_stats = compute_score_stats(file_path, max_matches)
    if verbose:
        print(f"  Score stats: mean={score_stats['mean']:.4f}, std={score_stats['std']:.4f}")
        print(f"  Score range: [{score_stats['min']:.2f}, {score_stats['max']:.2f}]")
    
    # 保存统计信息
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump({"score": score_stats, "gamma": gamma}, f, indent=2)
    
    # 第二遍：生成数据
    if verbose:
        print("Pass 2: Generating samples...")
    
    all_obs, all_vec, all_mask, all_act, all_value = [], [], [], [], []
    file_idx = 0
    match_count = 0
    sample_count = 0
    
    with open(file_path, encoding="utf-8") as f:
        match_lines = []
        in_match = False
        
        for line in f:
            t = line.split()
            if len(t) == 0:
                continue
                
            if t[0] == "Match":
                if in_match and match_lines:
                    # 处理上一局
                    samples = process_match(match_lines, gamma, score_stats)
                    for s in samples:
                        all_obs.append(s["obs"])
                        all_vec.append(s["vec"])
                        all_mask.append(s["mask"])
                        all_act.append(s["act"])
                        all_value.append(s["value_target"])
                    sample_count += len(samples)
                    
                    # 检查是否需要保存
                    if len(all_obs) >= samples_per_file:
                        _save_batch(output_dir, file_idx, all_obs, all_vec, all_mask, all_act, all_value)
                        if verbose:
                            print(f"  Saved {output_dir}/{file_idx}.npz ({len(all_obs)} samples)")
                        all_obs, all_vec, all_mask, all_act, all_value = [], [], [], [], []
                        file_idx += 1
                
                match_lines = [line]
                in_match = True
                match_count += 1
                
                if max_matches and match_count > max_matches:
                    break
            else:
                match_lines.append(line)
        
        # 处理最后一局
        if in_match and match_lines:
            samples = process_match(match_lines, gamma, score_stats)
            for s in samples:
                all_obs.append(s["obs"])
                all_vec.append(s["vec"])
                all_mask.append(s["mask"])
                all_act.append(s["act"])
                all_value.append(s["value_target"])
            sample_count += len(samples)
    
    # 保存剩余数据
    if all_obs:
        _save_batch(output_dir, file_idx, all_obs, all_vec, all_mask, all_act, all_value)
        if verbose:
            print(f"  Saved {output_dir}/{file_idx}.npz ({len(all_obs)} samples)")
    
    if verbose:
        print(f"\nDone! Total: {match_count} matches, {sample_count} samples, {file_idx + 1} files")
    
    return {"matches": match_count, "samples": sample_count, "files": file_idx + 1}


def _save_batch(output_dir, file_idx, obs, vec, mask, act, value):
    np.savez(
        os.path.join(output_dir, f"{file_idx}.npz"),
        obs=np.stack(obs).astype(np.int8),
        vec=np.stack(vec).astype(np.float16),
        mask=np.stack(mask).astype(np.int8),
        act=np.array(act, dtype=np.int16),
        value_target=np.array(value, dtype=np.float32),
    )


def process_range(
    file_path: str,
    output_dir: str,
    start_line: int,
    end_line: int,
    offset: int,
    cpu_id: int,
    gamma: float = 0.99,
    score_stats: Dict = None,
):
    """
    处理文件的指定行范围（并行模式使用）
    
    Args:
        file_path: 输入文件
        output_dir: 输出目录
        start_line: 起始行号（0-indexed）
        end_line: 结束行号（包含）
        offset: 起始 match 编号
        cpu_id: CPU ID（用于文件命名）
        gamma: 折扣因子
        score_stats: 预计算的分数统计（并行模式需要预先计算）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_obs, all_vec, all_mask, all_act, all_value = [], [], [], [], []
    match_count = 0
    sample_count = 0
    
    with open(file_path, encoding="utf-8") as f:
        # 跳过到起始行
        for _ in range(start_line):
            f.readline()
        
        match_lines = []
        in_match = False
        line_number = start_line
        
        for line in f:
            if line_number > end_line:
                break
            
            t = line.split()
            if len(t) == 0:
                line_number += 1
                continue
            
            if t[0] == "Match":
                if in_match and match_lines:
                    samples = process_match(match_lines, gamma, score_stats)
                    for s in samples:
                        all_obs.append(s["obs"])
                        all_vec.append(s["vec"])
                        all_mask.append(s["mask"])
                        all_act.append(s["act"])
                        all_value.append(s["value_target"])
                    sample_count += len(samples)
                
                match_lines = [line]
                in_match = True
                match_count += 1
            else:
                match_lines.append(line)
            
            line_number += 1
        
        # 处理最后一局
        if in_match and match_lines:
            samples = process_match(match_lines, gamma, score_stats)
            for s in samples:
                all_obs.append(s["obs"])
                all_vec.append(s["vec"])
                all_mask.append(s["mask"])
                all_act.append(s["act"])
                all_value.append(s["value_target"])
            sample_count += len(samples)
    
    # 保存数据
    if all_obs:
        _save_batch(output_dir, cpu_id, all_obs, all_vec, all_mask, all_act, all_value)
    
    # 保存统计
    with open(os.path.join(output_dir, f"count-{cpu_id}.json"), "w") as f:
        json.dump({"matches": match_count, "samples": sample_count}, f)
    
    print(f"[CPU {cpu_id}] Done: {match_count} matches, {sample_count} samples")
    return {"matches": match_count, "samples": sample_count}


def main():
    parser = argparse.ArgumentParser(description="Generate full training data for policy + value")
    parser.add_argument("--input", type=str, required=True, help="Input data.txt file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--samples_per_file", type=int, default=50000, help="Samples per output file")
    parser.add_argument("--max_matches", type=int, default=None, help="Max matches to process")
    # 并行模式参数
    parser.add_argument("--start_line", type=int, default=None, help="Start line (parallel mode)")
    parser.add_argument("--end_line", type=int, default=None, help="End line (parallel mode)")
    parser.add_argument("--offset", type=int, default=None, help="Match offset (parallel mode)")
    parser.add_argument("--cpu_id", type=int, default=None, help="CPU ID (parallel mode)")
    parser.add_argument("--score_mean", type=float, default=None, help="Pre-computed score mean")
    parser.add_argument("--score_std", type=float, default=None, help="Pre-computed score std")
    args = parser.parse_args()
    
    if args.start_line is not None:
        # 并行模式
        score_stats = {"mean": args.score_mean, "std": args.score_std}
        process_range(
            args.input,
            args.output,
            args.start_line,
            args.end_line,
            args.offset,
            args.cpu_id,
            gamma=args.gamma,
            score_stats=score_stats,
        )
    else:
        # 单进程模式
        process_file(
            args.input,
            args.output,
            gamma=args.gamma,
            samples_per_file=args.samples_per_file,
            max_matches=args.max_matches,
        )


if __name__ == "__main__":
    main()
