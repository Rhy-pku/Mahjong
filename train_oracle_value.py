#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an oracle teacher (policy + value) on full-information states.
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from model_pretrain import PretrainModel


def _iter_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "*.npz")))


def train(data_dir, device, hidden_dim, epochs, batch_size, lr, reward_scale, value_weight, save_path):
    files = _iter_files(data_dir)
    if not files:
        raise FileNotFoundError("No npz files found in %s" % data_dir)

    sample = np.load(files[0])
    in_channels = sample["oracle_obs"].shape[1]

    model = PretrainModel(hidden_dim=hidden_dim, use_vec=False, in_channels=in_channels).to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        total_count = 0
        for path in files:
            data = np.load(path)
            obs = data["oracle_obs"]
            actions = data["action"].astype(np.int64)
            rewards = data["reward"].astype(np.float32) / reward_scale
            idx = np.random.permutation(len(obs))
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start : start + batch_size]
                obs_t = torch.tensor(obs[batch_idx], dtype=torch.float32, device=device)
                act_t = torch.tensor(actions[batch_idx], dtype=torch.long, device=device)
                tgt_t = torch.tensor(rewards[batch_idx], dtype=torch.float32, device=device)
                logits, values = model(obs_t)
                policy_loss = F.cross_entropy(logits, act_t)
                value_loss = F.mse_loss(values.squeeze(-1), tgt_t)
                loss = policy_loss + value_weight * value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)
                total_count += len(batch_idx)
        avg_loss = total_loss / max(total_count, 1)
        print("epoch %d loss %.6f" % (epoch + 1, avg_loss))

    if save_path:
        torch.save(model.state_dict(), save_path)
        print("saved:", save_path)


def main():
    parser = argparse.ArgumentParser(description="Train oracle teacher (policy+value)")
    parser.add_argument("--data_dir", required=True, help="Directory with oracle npz files")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Model hidden dim")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Divide rewards by this value")
    parser.add_argument("--value_weight", type=float, default=0.1, help="Value loss weight")
    parser.add_argument("--save_path", required=True, help="Output checkpoint path")
    args = parser.parse_args()

    device = torch.device(args.device)
    train(
        data_dir=args.data_dir,
        device=device,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reward_scale=args.reward_scale,
        value_weight=args.value_weight,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
