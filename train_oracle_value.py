#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an oracle teacher (policy + value) on full-information states.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from model_pretrain import PretrainModel


def _iter_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "*.npz")))


def _apply_action_mask(logits, mask):
    inf_mask = torch.clamp(torch.log(mask + 1e-45), min=-1e38, max=0)
    return logits + inf_mask


class RandomBatchDataset(IterableDataset):
    def __init__(self, files, file_sizes, batch_size, reward_scale, use_reward_vec, seed=None):
        self.files = files
        self.file_sizes = file_sizes
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.use_reward_vec = use_reward_vec
        total = float(sum(file_sizes))
        self.weights = np.array([s / total for s in file_sizes], dtype=np.float64)
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        if self.seed is None:
            seed = int(time.time() * 1e6) ^ os.getpid()
        else:
            seed = self.seed
        if info is not None:
            seed = seed + info.id * 9973
        seed = seed % (2**32 - 1)
        rng = np.random.RandomState(seed)
        cache = {}
        try:
            while True:
                file_idx = rng.choice(len(self.files), p=self.weights)
                data = cache.get(file_idx)
                if data is None:
                    data = np.load(self.files[file_idx], mmap_mode="r")
                    cache[file_idx] = data
                n = data["oracle_obs"].shape[0]
                idx = rng.randint(0, n, size=self.batch_size)
                obs = data["oracle_obs"][idx]
                actions = data["action"][idx]
                masks = data["student_mask"][idx]
                players = data["player"][idx]
                rewards = data["reward"][idx].astype(np.float32) / self.reward_scale
                if self.use_reward_vec:
                    reward_vec = data["reward_vec"][idx].astype(np.float32) / self.reward_scale
                    yield obs, actions, masks, rewards, players, reward_vec
                else:
                    yield obs, actions, masks, rewards, players
        finally:
            for data in cache.values():
                try:
                    data.close()
                except Exception:
                    pass


def train(
    data_dir,
    device,
    hidden_dim,
    epochs,
    batch_size,
    lr,
    reward_scale,
    value_weight,
    value_dim,
    save_path,
    wandb_run=None,
    wandb_log_interval=50,
    progress_interval=50,
    num_workers=0,
    prefetch_factor=2,
    steps_per_epoch=None,
    seed=None,
):
    files = _iter_files(data_dir)
    if not files:
        raise FileNotFoundError("No npz files found in %s" % data_dir)

    sample = np.load(files[0])
    in_channels = sample["oracle_obs"].shape[1]

    model = PretrainModel(
        hidden_dim=hidden_dim,
        use_vec=False,
        in_channels=in_channels,
        value_dim=value_dim,
    ).to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    file_sizes = []
    total_samples = 0
    for path in files:
        with np.load(path, mmap_mode="r") as data:
            size = int(data["oracle_obs"].shape[0])
        file_sizes.append(size)
        total_samples += size
    total_batches = sum((size + batch_size - 1) // batch_size for size in file_sizes)
    if steps_per_epoch is None:
        steps_per_epoch = total_batches

    use_reward_vec = "reward_vec" in sample.files
    dataset = RandomBatchDataset(
        files=files,
        file_sizes=file_sizes,
        batch_size=batch_size,
        reward_scale=reward_scale,
        use_reward_vec=use_reward_vec,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=bool(num_workers),
    )

    for epoch in range(epochs):
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_count = 0
        batch_idx_global = 0
        pbar = None
        if tqdm is not None:
            pbar = tqdm(
                total=steps_per_epoch,
                desc="epoch %d/%d" % (epoch + 1, epochs),
                unit="batch",
                dynamic_ncols=True,
            )
        for batch in loader:
            if use_reward_vec:
                obs_np, actions_np, masks_np, rewards_np, players_np, reward_vec_np = batch
            else:
                obs_np, actions_np, masks_np, rewards_np, players_np = batch
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            act_t = torch.tensor(actions_np, dtype=torch.long, device=device)
            mask_t = torch.tensor(masks_np, dtype=torch.float32, device=device)
            tgt_t = torch.tensor(rewards_np, dtype=torch.float32, device=device)
            reward_vec_t = None
            if use_reward_vec:
                reward_vec_t = torch.tensor(reward_vec_np, dtype=torch.float32, device=device)
            logits, values = model(obs_t)
            masked_logits = _apply_action_mask(logits, mask_t)
            policy_loss = F.cross_entropy(masked_logits, act_t)
            if values.shape[-1] == 1:
                value_loss = F.mse_loss(values.squeeze(-1), tgt_t)
            else:
                if reward_vec_t is not None:
                    if values.shape[-1] != reward_vec_t.shape[-1]:
                        raise ValueError("value_dim %d != reward_vec dim %d" % (values.shape[-1], reward_vec_t.shape[-1]))
                    value_loss = F.mse_loss(values, reward_vec_t)
                else:
                    player_t = torch.tensor(players_np, dtype=torch.long, device=device)
                    pred = values.gather(1, player_t.unsqueeze(1)).squeeze(1)
                    value_loss = F.mse_loss(pred, tgt_t)
            loss = policy_loss + value_weight * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size_actual = obs_t.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_policy += policy_loss.item() * batch_size_actual
            total_value += value_loss.item() * batch_size_actual
            total_count += batch_size_actual

            if wandb_run and (batch_idx_global % wandb_log_interval == 0):
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        "train/policy_loss": policy_loss.item(),
                        "train/value_loss": value_loss.item(),
                        "epoch": epoch + 1,
                    }
                )
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    loss="%.4f" % loss.item(),
                    policy="%.4f" % policy_loss.item(),
                    value="%.4f" % value_loss.item(),
                )
            elif batch_idx_global % progress_interval == 0:
                print(
                    "epoch %d/%d batch %d/%d loss %.4f policy %.4f value %.4f"
                    % (
                        epoch + 1,
                        epochs,
                        batch_idx_global,
                        steps_per_epoch,
                        loss.item(),
                        policy_loss.item(),
                        value_loss.item(),
                    )
                )
            batch_idx_global += 1
            if batch_idx_global >= steps_per_epoch:
                break
        if pbar is not None:
            pbar.close()
        avg_loss = total_loss / max(total_count, 1)
        avg_policy = total_policy / max(total_count, 1)
        avg_value = total_value / max(total_count, 1)
        print(
            "epoch %d loss %.6f policy %.6f value %.6f"
            % (epoch + 1, avg_loss, avg_policy, avg_value)
        )
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "epoch/loss": avg_loss,
                    "epoch/policy_loss": avg_policy,
                    "epoch/value_loss": avg_value,
                }
            )

    if save_path:
        torch.save(model.state_dict(), save_path)
        print("saved:", save_path)


def main():
    parser = argparse.ArgumentParser(description="Train oracle teacher (policy+value)")
    parser.add_argument("--data_dir", required=True, help="Directory with oracle npz files")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Model hidden dim")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--reward_scale", type=float, default=100, help="Divide rewards by this value")
    parser.add_argument("--value_weight", type=float, default=1, help="Value loss weight")
    parser.add_argument("--value_dim", type=int, default=1, help="Value head output dim")
    parser.add_argument("--save_path", required=True, help="Output checkpoint path")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="mahjong-oracle", help="W&B project name")
    parser.add_argument("--wandb_run_name", default="", help="W&B run name")
    parser.add_argument("--wandb_log_interval", type=int, default=50, help="Log every N batches")
    parser.add_argument("--progress_interval", type=int, default=50, help="Print every N batches when tqdm is absent")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor for workers")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Override batches per epoch")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config={
                "hidden_dim": args.hidden_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "reward_scale": args.reward_scale,
                "value_weight": args.value_weight,
                "value_dim": args.value_dim,
            },
        )

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
        value_dim=args.value_dim,
        save_path=args.save_path,
        wandb_run=wandb_run,
        wandb_log_interval=args.wandb_log_interval,
        progress_interval=args.progress_interval,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
