#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train AlphaZero oracle (policy + value) from MCTS self-play data.
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
    def __init__(self, files, file_sizes, batch_size, seed=None):
        self.files = files
        self.file_sizes = file_sizes
        self.batch_size = batch_size
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
                masks = data["action_mask"][idx]
                pi = data["pi"][idx]
                rewards = data["reward_vec"][idx]
                yield obs, masks, pi, rewards
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
    value_weight,
    policy_weight,
    save_path,
    num_workers=0,
    prefetch_factor=2,
    steps_per_epoch=None,
    seed=None,
    wandb_run=None,
    wandb_log_interval=50,
    progress_interval=50,
    amp=False,
    dataloader_timeout=0,
):
    files = _iter_files(data_dir)
    if not files:
        raise FileNotFoundError("No npz files found in %s" % data_dir)

    sample = np.load(files[0])
    in_channels = sample["oracle_obs"].shape[1]
    value_dim = sample["reward_vec"].shape[1]

    model = PretrainModel(
        hidden_dim=hidden_dim,
        use_vec=False,
        in_channels=in_channels,
        value_dim=value_dim,
    ).to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

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

    print(
        "dataset: files=%d total_samples=%d batches=%d steps_per_epoch=%d"
        % (len(files), total_samples, total_batches, steps_per_epoch)
    )

    dataset = RandomBatchDataset(
        files=files,
        file_sizes=file_sizes,
        batch_size=batch_size,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=bool(num_workers),
        timeout=dataloader_timeout,
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
            obs_np, masks_np, pi_np, reward_np = batch
            obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32, non_blocking=True)
            mask_t = torch.from_numpy(masks_np).to(device=device, dtype=torch.float32, non_blocking=True)
            pi_t = torch.from_numpy(pi_np).to(device=device, dtype=torch.float32, non_blocking=True)
            reward_t = torch.from_numpy(reward_np).to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits, values = model(obs_t)
                masked_logits = _apply_action_mask(logits, mask_t)
                log_probs = F.log_softmax(masked_logits, dim=1)
                policy_loss = -(pi_t * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(values, reward_t)
                loss = policy_weight * policy_loss + value_weight * value_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
    parser = argparse.ArgumentParser(description="Train oracle with MCTS self-play data")
    parser.add_argument("--data_dir", required=True, help="Directory with self-play npz files")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Model hidden dim")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--value_weight", type=float, default=1.0, help="Value loss weight")
    parser.add_argument("--policy_weight", type=float, default=1.0, help="Policy loss weight")
    parser.add_argument("--save_path", required=True, help="Output checkpoint path")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor for workers")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Override batches per epoch")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--progress_interval", type=int, default=50, help="Print every N batches when tqdm is absent")
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument("--dataloader_timeout", type=int, default=0, help="DataLoader timeout (sec)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="mahjong-az", help="W&B project name")
    parser.add_argument("--wandb_run_name", default="", help="W&B run name")
    parser.add_argument("--wandb_log_interval", type=int, default=50, help="Log every N batches")
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
                "value_weight": args.value_weight,
                "policy_weight": args.policy_weight,
                "amp": args.amp,
                "num_workers": args.num_workers,
            },
        )

    device = torch.device(args.device)
    print(
        "train config: data_dir=%s device=%s epochs=%d batch=%d lr=%.6f workers=%d"
        % (
            args.data_dir,
            args.device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.num_workers,
        )
    )
    train(
        data_dir=args.data_dir,
        device=device,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_weight=args.value_weight,
        policy_weight=args.policy_weight,
        save_path=args.save_path,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        wandb_run=wandb_run,
        wandb_log_interval=args.wandb_log_interval,
        progress_interval=args.progress_interval,
        amp=args.amp,
        dataloader_timeout=args.dataloader_timeout,
    )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
