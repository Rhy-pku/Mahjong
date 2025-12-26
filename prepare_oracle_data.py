#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build paired oracle/student datasets from full-information logs.

Outputs npz files with:
- oracle_obs: (N, C_oracle, 4, 9) int8
- student_obs: (N, 60, 4, 9) int8
- student_vec: (N, 117) float16
- student_mask: (N, 235) int8
- action: (N,) int16
- reward: (N,) float32
- player: (N,) int8
- match_id: (N,) int64
- step_id: (N,) int32
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np

from feature_10m import FeatureAgent10M as FeatureAgent

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


def _encode_counts(obs, channel, tiles):
    for tile in tiles:
        rc = _tile_rc(tile)
        if rc is None:
            continue
        r, c = rc
        obs[channel, r, c] = min(obs[channel, r, c] + 1, 4)


class OracleState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prevalent_wind = 0
        self.hands = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.packs = [[] for _ in range(4)]
        self.remaining = {t: 4 for t in TILE_LIST}
        self.last_discard_tile = None
        self.last_discard_player = None

    def set_wind(self, wind):
        self.prevalent_wind = wind

    def deal(self, player, tiles):
        self.hands[player] = list(tiles)
        for t in tiles:
            if t in self.remaining:
                self.remaining[t] -= 1

    def draw(self, player, tile):
        self.hands[player].append(tile)
        if tile in self.remaining:
            self.remaining[tile] -= 1

    def play(self, player, tile):
        if tile in self.hands[player]:
            self.hands[player].remove(tile)
        self.discards[player].append(tile)
        self.last_discard_tile = tile
        self.last_discard_player = player

    def chi(self, player, mid_tile):
        if self.last_discard_tile is not None and self.last_discard_player is not None:
            if self.last_discard_tile in self.discards[self.last_discard_player]:
                self.discards[self.last_discard_player].remove(self.last_discard_tile)
        color = mid_tile[0]
        num = int(mid_tile[1])
        seq = [color + str(num - 1), color + str(num), color + str(num + 1)]
        for t in seq:
            if t != self.last_discard_tile and t in self.hands[player]:
                self.hands[player].remove(t)
        self.packs[player].append(("CHI", seq))

    def peng(self, player, tile):
        if self.last_discard_tile is not None and self.last_discard_player is not None:
            if self.last_discard_tile in self.discards[self.last_discard_player]:
                self.discards[self.last_discard_player].remove(self.last_discard_tile)
        removed = 0
        while tile in self.hands[player] and removed < 2:
            self.hands[player].remove(tile)
            removed += 1
        self.packs[player].append(("PENG", [tile] * 3))

    def gang(self, player, tile):
        if self.last_discard_tile is not None and self.last_discard_player is not None:
            if self.last_discard_tile in self.discards[self.last_discard_player]:
                self.discards[self.last_discard_player].remove(self.last_discard_tile)
        removed = 0
        while tile in self.hands[player] and removed < 3:
            self.hands[player].remove(tile)
            removed += 1
        self.packs[player].append(("GANG", [tile] * 4))

    def angang(self, player, tile):
        removed = 0
        while tile in self.hands[player] and removed < 4:
            self.hands[player].remove(tile)
            removed += 1
        self.packs[player].append(("GANG", [tile] * 4))

    def bugang(self, player, tile):
        for i, pack in enumerate(self.packs[player]):
            if pack[0] == "PENG" and pack[1] and pack[1][0] == tile:
                self.packs[player][i] = ("GANG", [tile] * 4)
                break
        if tile in self.hands[player]:
            self.hands[player].remove(tile)

    def make_teacher_obs(self, player, student_obs):
        extra_hand_channels = 12
        extra_remaining_channels = 34
        out = np.zeros((STUDENT_CHANNELS + extra_hand_channels + extra_remaining_channels, 4, 9), dtype=np.int8)
        out[:STUDENT_CHANNELS] = student_obs

        order = [player, (player + 1) % 4, (player + 2) % 4, (player + 3) % 4]
        offset = STUDENT_CHANNELS
        for pid in order[1:]:
            _encode_hand(out, offset, self.hands[pid])
            offset += 4

        for tile_idx, tile in enumerate(TILE_LIST):
            r, c = divmod(tile_idx, 9)
            out[offset + tile_idx, r, c] = max(self.remaining.get(tile, 0), 0)

        return out


def _add_sample(player, oracle_state, obs_dict, player_samples, match_id, step_id):
    if obs_dict is None:
        return
    student_obs = obs_dict["observation"].copy()
    sample = {
        "oracle_obs": oracle_state.make_teacher_obs(player, student_obs),
        "student_obs": student_obs,
        "student_vec": obs_dict["vec"].copy(),
        "student_mask": obs_dict["action_mask"].copy(),
        "action": 0,
        "player": player,
        "match_id": match_id,
        "step_id": step_id,
    }
    player_samples[player].append(sample)


def process_match(lines: List[str]) -> List[dict]:
    oracle = OracleState()
    agents = [FeatureAgent(i) for i in range(4)]
    player_samples = [[] for _ in range(4)]
    cur_tile = None
    scores = None
    step_id = 0
    match_id = "0"

    if lines:
        first = lines[0].split()
        if len(first) >= 2 and first[0] == "Match":
            match_id = first[1]

    for line in lines:
        t = line.split()
        if not t:
            continue
        if t[0] == "Wind":
            oracle.set_wind(int(t[1]))
            for agent in agents:
                agent.request2obs(line)
            continue
        if t[0] == "Huang":
            for agent in agents:
                agent.request2obs(line)
            continue
        if t[0] == "Score":
            scores = [float(x) for x in t[1:5]]
            continue
        if t[0] == "Player":
            p = int(t[1])
            action = t[2]
            if action == "Deal":
                tiles = t[3:]
                oracle.deal(p, tiles)
                agents[p].request2obs(" ".join(t[2:]))
            elif action == "Draw":
                tile = t[3]
                oracle.draw(p, tile)
                obs_dict = agents[p].request2obs(" ".join(t[2:]))
                _add_sample(p, oracle, obs_dict, player_samples, match_id, step_id)
                step_id += 1
                for i in range(4):
                    if i != p:
                        agents[i].request2obs(" ".join(t[:3]))
            elif action == "Play":
                cur_tile = t[3]
                if player_samples[p]:
                    act = agents[p].response2action(" ".join(t[2:]))
                    player_samples[p][-1]["action"] = act
                oracle.play(p, cur_tile)
                agents[p].request2obs(line)
                for i in range(4):
                    if i != p:
                        obs_dict = agents[i].request2obs(line)
                        _add_sample(i, oracle, obs_dict, player_samples, match_id, step_id)
                        step_id += 1
            elif action == "Chi":
                if player_samples[p]:
                    act = agents[p].response2action("Chi %s %s" % (cur_tile, t[3]))
                    player_samples[p][-1]["action"] = act
                oracle.chi(p, t[3])
                for i in range(4):
                    if i == p:
                        obs_dict = agents[p].request2obs("Player %d Chi %s" % (p, t[3]))
                        _add_sample(p, oracle, obs_dict, player_samples, match_id, step_id)
                        step_id += 1
                    else:
                        agents[i].request2obs("Player %d Chi %s" % (p, t[3]))
            elif action == "Peng":
                if player_samples[p]:
                    act = agents[p].response2action("Peng %s" % t[3])
                    player_samples[p][-1]["action"] = act
                oracle.peng(p, t[3])
                for i in range(4):
                    if i == p:
                        obs_dict = agents[p].request2obs("Player %d Peng %s" % (p, t[3]))
                        _add_sample(p, oracle, obs_dict, player_samples, match_id, step_id)
                        step_id += 1
                    else:
                        agents[i].request2obs("Player %d Peng %s" % (p, t[3]))
            elif action == "Gang":
                if player_samples[p]:
                    act = agents[p].response2action("Gang %s" % t[3])
                    player_samples[p][-1]["action"] = act
                tile = t[3] if len(t) > 3 else cur_tile
                oracle.gang(p, tile)
                for i in range(4):
                    agents[i].request2obs("Player %d Gang %s" % (p, tile))
            elif action == "AnGang":
                if player_samples[p]:
                    act = agents[p].response2action("AnGang %s" % t[3])
                    player_samples[p][-1]["action"] = act
                tile = t[3] if len(t) > 3 else cur_tile
                oracle.angang(p, tile)
                for i in range(4):
                    if i == p:
                        agents[p].request2obs("Player %d AnGang %s" % (p, tile))
                    else:
                        agents[i].request2obs("Player %d AnGang" % p)
            elif action == "BuGang":
                if player_samples[p]:
                    act = agents[p].response2action("BuGang %s" % t[3])
                    player_samples[p][-1]["action"] = act
                oracle.bugang(p, t[3])
                for i in range(4):
                    if i == p:
                        agents[p].request2obs("Player %d BuGang %s" % (p, t[3]))
                    else:
                        obs_dict = agents[i].request2obs("Player %d BuGang %s" % (p, t[3]))
                        _add_sample(i, oracle, obs_dict, player_samples, match_id, step_id)
                        step_id += 1
            elif action == "Hu":
                if player_samples[p]:
                    act = agents[p].response2action("Hu")
                    player_samples[p][-1]["action"] = act
                for i in range(4):
                    agents[i].request2obs("Player %d Hu" % p)

            if action in ["Peng", "Gang", "Hu"]:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p2 = int(t[k + 1])
                        if t[k + 2] == "Chi" and player_samples[p2]:
                            act = agents[p2].response2action("Chi %s %s" % (cur_tile, t[k + 3]))
                            player_samples[p2][-1]["action"] = act
                        elif t[k + 2] == "Peng" and player_samples[p2]:
                            act = agents[p2].response2action("Peng %s" % t[k + 3])
                            player_samples[p2][-1]["action"] = act
                        elif t[k + 2] == "Gang" and player_samples[p2]:
                            act = agents[p2].response2action("Gang %s" % t[k + 3])
                            player_samples[p2][-1]["action"] = act
                        elif t[k + 2] == "Hu" and player_samples[p2]:
                            act = agents[p2].response2action("Hu")
                            player_samples[p2][-1]["action"] = act
                    else:
                        break

    if scores is None:
        return []
    all_samples = []
    for p in range(4):
        for s in player_samples[p]:
            s["reward"] = scores[p]
            s["reward_vec"] = scores
            all_samples.append(s)
    return all_samples


def _save_batch(output_dir, file_idx, samples):
    np.savez(
        os.path.join(output_dir, f"{file_idx}.npz"),
        oracle_obs=np.stack([s["oracle_obs"] for s in samples]).astype(np.int8),
        student_obs=np.stack([s["student_obs"] for s in samples]).astype(np.int8),
        student_vec=np.stack([s["student_vec"] for s in samples]).astype(np.float16),
        student_mask=np.stack([s["student_mask"] for s in samples]).astype(np.int8),
        action=np.array([s["action"] for s in samples], dtype=np.int16),
        reward=np.array([s["reward"] for s in samples], dtype=np.float32),
        reward_vec=np.stack([s["reward_vec"] for s in samples]).astype(np.float32),
        player=np.array([s["player"] for s in samples], dtype=np.int8),
        match_id=np.array([s["match_id"] for s in samples], dtype="U32"),
        step_id=np.array([s["step_id"] for s in samples], dtype=np.int32),
    )


def _read_matches(file_path, max_matches=None):
    matches = []
    with open(file_path, encoding="utf-8") as f:
        match_lines = []
        in_match = False
        match_count = 0
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "Match":
                if in_match and match_lines:
                    matches.append(match_lines)
                match_lines = [line]
                in_match = True
                match_count += 1
                if max_matches and match_count > max_matches:
                    break
            else:
                match_lines.append(line)
        if in_match and match_lines and (not max_matches or match_count <= max_matches):
            matches.append(match_lines)
    return matches


def process_file(file_path, output_dir, samples_per_file=50000, max_matches=None, verbose=True, workers=1):
    os.makedirs(output_dir, exist_ok=True)
    file_idx = 0
    samples = []

    if workers > 1:
        matches = _read_matches(file_path, max_matches=max_matches)
        with Pool(processes=workers) as pool:
            for new_samples in pool.imap_unordered(process_match, matches, chunksize=1):
                samples.extend(new_samples)
                if len(samples) >= samples_per_file:
                    _save_batch(output_dir, file_idx, samples)
                    if verbose:
                        print(f"  Saved {output_dir}/{file_idx}.npz ({len(samples)} samples)")
                    samples = []
                    file_idx += 1
    else:
        with open(file_path, encoding="utf-8") as f:
            match_lines = []
            in_match = False
            match_count = 0
            for line in f:
                t = line.split()
                if not t:
                    continue
                if t[0] == "Match":
                    if in_match and match_lines:
                        new_samples = process_match(match_lines)
                        samples.extend(new_samples)
                        if len(samples) >= samples_per_file:
                            _save_batch(output_dir, file_idx, samples)
                            if verbose:
                                print(f"  Saved {output_dir}/{file_idx}.npz ({len(samples)} samples)")
                            samples = []
                            file_idx += 1
                    match_lines = [line]
                    in_match = True
                    match_count += 1
                    if max_matches and match_count > max_matches:
                        break
                else:
                    match_lines.append(line)

            if in_match and match_lines and (not max_matches or match_count <= max_matches):
                new_samples = process_match(match_lines)
                samples.extend(new_samples)

    if samples:
        _save_batch(output_dir, file_idx, samples)
        if verbose:
            print(f"  Saved {output_dir}/{file_idx}.npz ({len(samples)} samples)")

    if verbose:
        print("Done! Files:", file_idx + 1)


def main():
    parser = argparse.ArgumentParser(description="Generate oracle/student paired datasets")
    parser.add_argument("--input", required=True, help="Input log file (data.txt)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--samples_per_file", type=int, default=50000, help="Samples per output file")
    parser.add_argument("--max_matches", type=int, default=None, help="Max matches to process")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    process_file(
        args.input,
        args.output,
        samples_per_file=args.samples_per_file,
        max_matches=args.max_matches,
        workers=max(args.workers, 1),
    )


if __name__ == "__main__":
    main()
