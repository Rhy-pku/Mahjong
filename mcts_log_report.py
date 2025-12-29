#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize MCTS debug JSONL logs and optionally render DOT trees to PNG.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from collections import Counter


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _entropy(probs):
    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0:
        return 0.0
    ent = 0.0
    for p in probs:
        p = p / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


def _argmax_index(values):
    if not values:
        return None
    best_idx = 0
    best_val = values[0]
    for i, v in enumerate(values):
        if v > best_val:
            best_idx = i
            best_val = v
    return best_idx


def _extract_root_stats(obj):
    valid = obj.get("root_valid")
    prior_valid = obj.get("root_prior_valid")
    nsa_valid = obj.get("root_nsa_valid")
    if valid is None:
        prior = obj.get("root_prior")
        nsa = obj.get("root_nsa")
        if isinstance(prior, list):
            valid = list(range(len(prior)))
            prior_valid = prior
        if isinstance(nsa, list):
            nsa_valid = nsa
    return valid, prior_valid, nsa_valid


def _parse_tree_files(trees_dir):
    if not trees_dir:
        return {}
    if not os.path.isdir(trees_dir):
        return {}
    mapping = {}
    for name in os.listdir(trees_dir):
        if not name.endswith(".dot"):
            continue
        parts = name.split("_")
        step = None
        for part in parts:
            if part.startswith("step"):
                try:
                    step = int(part.replace("step", "").replace(".dot", ""))
                except ValueError:
                    step = None
                break
        if step is not None:
            mapping[step] = os.path.join(trees_dir, name)
    return mapping


def _render_trees(tree_map, overwrite=False):
    if not tree_map:
        return {}
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return {}
    png_map = {}
    for step, dot_path in tree_map.items():
        png_path = dot_path[:-4] + ".png"
        if os.path.exists(png_path) and not overwrite:
            png_map[step] = png_path
            continue
        try:
            subprocess.run([dot_bin, "-Tpng", dot_path, "-o", png_path], check=True)
            png_map[step] = png_path
        except subprocess.SubprocessError:
            continue
    return png_map


def summarize_log(log_path, trees_dir=None, render_trees=False, overwrite_trees=False):
    counts = Counter()
    decision_rows = []
    u_dom = 0
    q_dom = 0
    u_abs_sum = 0.0
    q_abs_sum = 0.0
    mcts_select_steps = 0
    mcts_decisions = 0
    mcts_argmax_diff = 0

    tree_map = _parse_tree_files(trees_dir)
    png_map = _render_trees(tree_map, overwrite=overwrite_trees) if render_trees else {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("type")
            counts[t] += 1

            if t == "decision_end" and obj.get("use_mcts"):
                mcts_decisions += 1
                step = obj.get("step", -1)
                valid, prior_valid, nsa_valid = _extract_root_stats(obj)
                action = obj.get("action")
                action_is_argmax = None
                prior_max = None
                prior_second = None
                prior_entropy = None
                explored = None
                valid_count = None
                argmax_action = None
                if isinstance(valid, list):
                    valid_count = len(valid)
                if isinstance(prior_valid, list) and prior_valid:
                    prior_max = max(prior_valid)
                    sorted_pri = sorted(prior_valid, reverse=True)
                    prior_second = sorted_pri[1] if len(sorted_pri) > 1 else None
                    prior_entropy = _entropy(prior_valid)
                    arg_idx = _argmax_index(prior_valid)
                    if arg_idx is not None and isinstance(valid, list) and arg_idx < len(valid):
                        argmax_action = valid[arg_idx]
                        action_is_argmax = (argmax_action == action)
                        if not action_is_argmax:
                            mcts_argmax_diff += 1
                if isinstance(nsa_valid, list):
                    explored = sum(1 for v in nsa_valid if v > 0)
                decision_rows.append(
                    {
                        "step": step,
                        "action": action,
                        "argmax_action": argmax_action,
                        "action_is_argmax": action_is_argmax,
                        "root_n_visits": obj.get("root_n_visits"),
                        "root_valid_count": valid_count,
                        "root_explored_branches": explored,
                        "root_prior_max": prior_max,
                        "root_prior_second": prior_second,
                        "root_prior_entropy": prior_entropy,
                        "tree_dot": tree_map.get(step),
                        "tree_png": png_map.get(step),
                    }
                )

            if t in ("leaf_eval", "simulation_terminal"):
                steps = obj.get("steps")
                if not isinstance(steps, list):
                    continue
                for step in steps:
                    if step.get("policy") != "mcts_select":
                        continue
                    valid = step.get("valid_actions")
                    q_vals = step.get("q_vals")
                    u_vals = step.get("u_vals")
                    action = step.get("chosen_action")
                    if not (isinstance(valid, list) and isinstance(q_vals, list) and isinstance(u_vals, list)):
                        continue
                    try:
                        idx = valid.index(action)
                    except ValueError:
                        continue
                    q = _safe_float(q_vals[idx], 0.0)
                    u = _safe_float(u_vals[idx], 0.0)
                    mcts_select_steps += 1
                    if abs(u) >= abs(q):
                        u_dom += 1
                    else:
                        q_dom += 1
                    u_abs_sum += abs(u)
                    q_abs_sum += abs(q)

    summary = {
        "type_counts": dict(counts),
        "mcts_decisions": mcts_decisions,
        "mcts_action_diff_count": mcts_argmax_diff,
        "mcts_action_diff_ratio": (mcts_argmax_diff / mcts_decisions) if mcts_decisions else 0.0,
        "mcts_select_steps": mcts_select_steps,
        "u_dom_ratio": (u_dom / mcts_select_steps) if mcts_select_steps else 0.0,
        "avg_abs_u": (u_abs_sum / mcts_select_steps) if mcts_select_steps else 0.0,
        "avg_abs_q": (q_abs_sum / mcts_select_steps) if mcts_select_steps else 0.0,
        "trees_found": len(tree_map),
        "trees_rendered": len(png_map),
    }
    return summary, decision_rows


def write_csv(csv_path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(md_path, summary, rows):
    lines = []
    lines.append("# MCTS Debug Report")
    lines.append("")
    lines.append("## Summary")
    for key, val in summary.items():
        lines.append("- %s: %s" % (key, val))
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    header = [
        "step",
        "action",
        "argmax_action",
        "action_is_argmax",
        "root_n_visits",
        "root_valid_count",
        "root_explored_branches",
        "root_prior_max",
        "root_prior_second",
        "root_prior_entropy",
        "tree_dot",
        "tree_png",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        values = [row.get(h) for h in header]
        safe = ["" if v is None else str(v) for v in values]
        lines.append("| " + " | ".join(safe) + " |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Summarize MCTS debug JSONL logs")
    parser.add_argument("--log", required=True, help="Path to JSONL log file")
    parser.add_argument("--trees", default="", help="Path to trees dir with DOT files")
    parser.add_argument("--render-trees", action="store_true", help="Render DOT files to PNG using graphviz")
    parser.add_argument("--overwrite-trees", action="store_true", help="Overwrite existing PNG files")
    parser.add_argument("--out", default="", help="Write markdown report to this path")
    parser.add_argument("--csv", default="", help="Write CSV report to this path")
    args = parser.parse_args()

    summary, rows = summarize_log(
        args.log,
        trees_dir=args.trees.strip() or None,
        render_trees=args.render_trees,
        overwrite_trees=args.overwrite_trees,
    )

    print("summary:", summary)
    print("decisions:", len(rows))

    if args.csv:
        write_csv(args.csv, rows)
        print("csv:", args.csv)
    if args.out:
        write_md(args.out, summary, rows)
        print("md:", args.out)


if __name__ == "__main__":
    main()
