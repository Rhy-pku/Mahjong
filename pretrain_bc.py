#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mahjong BC pretrain (cross-entropy) — streaming / GPU-first paradigm.

Key ideas:
- IterableDataset that shards files across (DDP ranks) and (DataLoader workers)
- Each worker loads a whole .npz once, then yields *mini-batches* (not samples)
- Optional GPU prefetch on a dedicated CUDA stream to overlap H2D copy and compute
- Vectorized accuracy / topk metrics on GPU (no Python loops)
- Optional torch.compile + AMP + fused AdamW (if available)

Training objective remains identical to original: CE(logits, act).
"""

import os
import re
import json
import math
import argparse
import contextlib
from glob import glob
from dataclasses import dataclass
from typing import Iterator, Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

from model_pretrain import PretrainModel


# -------------------------
# Distributed helpers
# -------------------------
def ddp_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def ddp_rank_world() -> Tuple[int, int]:
    if ddp_is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1

def ddp_barrier():
    if ddp_is_initialized():
        torch.distributed.barrier()

def ddp_allreduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_initialized():
        return x
    x = x.clone()
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    x /= torch.distributed.get_world_size()
    return x


# -------------------------
# Iterable dataset: shard by files, yield batches
# -------------------------
@dataclass
class Batch:
    obs: torch.Tensor
    vec: torch.Tensor
    mask: torch.Tensor
    act: torch.Tensor

class NPZShardBatchDataset(IterableDataset):
    """
    Streams .npz shards and yields pre-batched tensors on CPU (optionally pinned).
    Each worker loads each file once, then yields batches by slicing arrays.
    """

    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "*.npz",
        batch_size: int = 1024,
        shuffle_files: bool = True,
        shuffle_in_file: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        pin_memory: bool = True,
        limit_files: Optional[int] = None,
    ):
        super().__init__()
        self.files = sorted(glob(os.path.join(data_dir, file_pattern)))
        if not self.files:
            raise ValueError(f"No data files found: {data_dir}/{file_pattern}")
        if limit_files is not None:
            self.files = self.files[: int(limit_files)]

        self.batch_size = int(batch_size)
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_in_file = bool(shuffle_in_file)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.pin_memory = bool(pin_memory)

    def _assigned_files(self) -> List[str]:
        # Shard across DDP ranks and DataLoader workers by file index
        rank, world = ddp_rank_world()
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1

        # global shard id among (world * num_workers)
        shard_id = rank * num_workers + worker_id
        shard_count = world * num_workers

        files = self.files[shard_id::shard_count]
        return files

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = self._assigned_files()
        # Per-iterator RNG (important: different workers, different order)
        rank, world = ddp_rank_world()
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        rng = np.random.default_rng(self.seed + 1000 * rank + 10 * worker_id)

        if self.shuffle_files:
            rng.shuffle(files)

        for fp in files:
            data = np.load(fp, allow_pickle=False)
            obs = data["obs"]
            vec = data["vec"]
            mask = data["mask"]
            act = data["act"]

            n = int(act.shape[0])
            if n <= 0:
                continue

            idx = np.arange(n)
            if self.shuffle_in_file:
                rng.shuffle(idx)

            bs = self.batch_size
            end = n - (n % bs) if self.drop_last else n

            # yield batches
            for s in range(0, end, bs):
                sl = idx[s:s+bs]
                # Slice numpy -> torch on CPU
                b_obs = torch.from_numpy(obs[sl])
                b_vec = torch.from_numpy(vec[sl])
                b_mask = torch.from_numpy(mask[sl])
                b_act = torch.as_tensor(act[sl], dtype=torch.long)

                yield {"obs": b_obs, "vec": b_vec, "mask": b_mask, "act": b_act}


# -------------------------
# GPU prefetcher (overlap H2D copy with compute)
# -------------------------
class CUDAPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.iter = None
        self.next_batch = None

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Batch:
        # non_blocking requires pinned memory to be effective
        obs = batch["obs"].to(self.device, non_blocking=True)
        vec = batch["vec"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        act = batch["act"].to(self.device, non_blocking=True)
        return Batch(obs=obs, vec=vec, mask=mask, act=act)

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device(batch)

    def __next__(self) -> Batch:
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch


# -------------------------
# Metrics (GPU vectorized)
# -------------------------
@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    # logits: [B, A], target: [B]
    pred = logits.argmax(dim=1)
    acc1 = (pred == target).float().mean()

    topk = logits.topk(k, dim=1).indices  # [B, k]
    # broadcast compare: [B, k] == [B, 1]
    acck = (topk == target.unsqueeze(1)).any(dim=1).float().mean()
    return acc1, acck


# -------------------------
# Train / Val steps
# -------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_accum: int,
    prefetch_to_gpu: bool,
    max_grad_norm: float,
    log_interval: int,
) -> Dict[str, float]:
    model.train()

    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else contextlib.nullcontext
    autocast_kwargs = {"enabled": use_amp} if device.type == "cuda" else {}

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_acc1 = 0.0
    total_samples = 0

    step = 0
    if prefetch_to_gpu and device.type == "cuda":
        it = CUDAPrefetcher(loader, device)
    else:
        it = iter(loader)

    is_rank0 = (ddp_rank_world()[0] == 0)
    if TQDM_AVAILABLE and is_rank0:
        try:
            total = len(loader)
        except TypeError:
            total = None
        it = tqdm(it, total=total, desc="train", leave=False)

    for batch in it:
        if isinstance(batch, dict):
            # non-prefetch path: move here
            obs = batch["obs"].to(device, non_blocking=(device.type == "cuda"))
            vec = batch["vec"].to(device, non_blocking=(device.type == "cuda"))
            mask = batch["mask"].to(device, non_blocking=(device.type == "cuda"))
            act = batch["act"].to(device, non_blocking=(device.type == "cuda"))
        else:
            # prefetch path: already Batch on device
            obs, vec, mask, act = batch.obs, batch.vec, batch.mask, batch.act

        bs = act.shape[0]

        with autocast_ctx(**autocast_kwargs):
            logits, _ = model(obs, vec, mask)
            loss = F.cross_entropy(logits, act)
            loss_scaled = loss / grad_accum

        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum == 0:
            if device.type == "cuda":
                scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            acc1, _ = accuracy_topk(logits, act, k=5)

        total_loss += float(loss.detach()) * bs
        total_acc1 += float(acc1.detach()) * bs
        total_samples += bs

        if log_interval > 0 and (step + 1) % log_interval == 0:
            # lightweight progress print (rank0 only)
            if is_rank0:
                msg = f"  step {step+1}: loss={total_loss/total_samples:.4f} acc={total_acc1/total_samples:.4f}"
                if TQDM_AVAILABLE:
                    tqdm.write(msg)
                else:
                    print(msg)

        step += 1

    # DDP mean across ranks
    loss_t = torch.tensor(total_loss / max(total_samples, 1), device=device, dtype=torch.float32)
    acc_t = torch.tensor(total_acc1 / max(total_samples, 1), device=device, dtype=torch.float32)
    loss_t = ddp_allreduce_mean(loss_t)
    acc_t = ddp_allreduce_mean(acc_t)

    return {"loss": float(loss_t.item()), "accuracy": float(acc_t.item())}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    prefetch_to_gpu: bool,
) -> Dict[str, float]:
    model.eval()

    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else contextlib.nullcontext
    autocast_kwargs = {"enabled": use_amp} if device.type == "cuda" else {}

    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_samples = 0

    if prefetch_to_gpu and device.type == "cuda":
        it = CUDAPrefetcher(loader, device)
    else:
        it = iter(loader)

    for batch in it:
        if isinstance(batch, dict):
            obs = batch["obs"].to(device, non_blocking=(device.type == "cuda"))
            vec = batch["vec"].to(device, non_blocking=(device.type == "cuda"))
            mask = batch["mask"].to(device, non_blocking=(device.type == "cuda"))
            act = batch["act"].to(device, non_blocking=(device.type == "cuda"))
        else:
            obs, vec, mask, act = batch.obs, batch.vec, batch.mask, batch.act

        bs = act.shape[0]
        with autocast_ctx(**autocast_kwargs):
            logits, _ = model(obs, vec, mask)
            loss = F.cross_entropy(logits, act)

        acc1, acc5 = accuracy_topk(logits, act, k=5)

        total_loss += float(loss.detach()) * bs
        total_acc1 += float(acc1.detach()) * bs
        total_acc5 += float(acc5.detach()) * bs
        total_samples += bs

    loss_t = torch.tensor(total_loss / max(total_samples, 1), device=device, dtype=torch.float32)
    acc1_t = torch.tensor(total_acc1 / max(total_samples, 1), device=device, dtype=torch.float32)
    acc5_t = torch.tensor(total_acc5 / max(total_samples, 1), device=device, dtype=torch.float32)

    loss_t = ddp_allreduce_mean(loss_t)
    acc1_t = ddp_allreduce_mean(acc1_t)
    acc5_t = ddp_allreduce_mean(acc5_t)

    return {"loss": float(loss_t.item()), "accuracy": float(acc1_t.item()), "top5_accuracy": float(acc5_t.item())}


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Mahjong BC streaming trainer (GPU-first)")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--val_data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./checkpoints_stream")

    p.add_argument("--model_type", type=str, default="full", choices=["full", "light"])
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--use_vec", action="store_true", default=True)

    # Data
    p.add_argument("--file_pattern", type=str, default="*.npz")
    p.add_argument("--batch_size", type=int, default=2048, help="GLOBAL per-process batch size (IterableDataset already yields batches)")
    p.add_argument("--val_batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--shuffle_files", action="store_true", default=True)
    p.add_argument("--no_shuffle_in_file", action="store_true", help="Disable in-file shuffle")
    p.add_argument("--drop_last", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit_files", type=int, default=None)

    # Train
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # GPU perf
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--prefetch_to_gpu", action="store_true", default=True)
    p.add_argument("--no_prefetch_to_gpu", action="store_true", help="Disable GPU prefetcher")
    p.add_argument("--tf32", action="store_true", default=True)

    # DDP
    p.add_argument("--ddp", action="store_true", help="Use torch.distributed (launch with torchrun)")
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

    # Log / ckpt
    p.add_argument("--save_interval", type=int, default=1)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_interval", type=int, default=50)

    # wandb
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mahjong-pretrain")
    p.add_argument("--wandb_name", type=str, default=None)

    return p.parse_args()


def maybe_init_ddp(args):
    if not args.ddp:
        return

    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed not available")
    if torch.distributed.is_initialized():
        return

    # torchrun sets these envs
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)


def main():
    args = parse_args()
    maybe_init_ddp(args)

    rank, world = ddp_rank_world()
    is_rank0 = (rank == 0)

    os.makedirs(args.output_dir, exist_ok=True)
    if is_rank0:
        with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    # Perf knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # wandb (rank0 only)
    if WANDB_AVAILABLE and (not args.no_wandb) and is_rank0:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # Dataset / Loader (NOTE: dataset yields batches => DataLoader batch_size=None)
    pin_memory = (device.type == "cuda")
    train_ds = NPZShardBatchDataset(
        data_dir=args.data_dir,
        file_pattern=args.file_pattern,
        batch_size=args.batch_size,
        shuffle_files=args.shuffle_files,
        shuffle_in_file=(not args.no_shuffle_in_file),
        drop_last=args.drop_last,
        seed=args.seed,
        pin_memory=pin_memory,
        limit_files=args.limit_files,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

    val_loader = None
    if args.val_data_dir:
        vbs = args.val_batch_size or args.batch_size
        val_ds = NPZShardBatchDataset(
            data_dir=args.val_data_dir,
            file_pattern=args.file_pattern,
            batch_size=vbs,
            shuffle_files=False,
            shuffle_in_file=False,
            drop_last=False,
            seed=args.seed + 999,
            pin_memory=pin_memory,
            limit_files=args.limit_files,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=None,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        )

    # Model
    if args.model_type == "full":
        model = PretrainModel(hidden_dim=args.hidden_dim, use_vec=args.use_vec)
    else:
        model = LightPretrainModel(hidden_dim=args.hidden_dim)
    model = model.to(device)

    if args.compile and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)

    if args.ddp and device.type == "cuda":
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
        )
    elif args.ddp and device.type != "cuda":
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Optimizer (fused if available)
    optim_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs, fused=(device.type == "cuda"))
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        map_loc = {"cuda:%d" % 0: "cuda:%d" % args.local_rank} if device.type == "cuda" else device
        ckpt = torch.load(args.resume, map_location=map_loc)
        state = ckpt["model"]
        # If saved from DDP wrapper, load accordingly
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_acc = float(ckpt.get("best_val_acc", 0.0))
        if is_rank0:
            print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")

    prefetch_to_gpu = bool(args.prefetch_to_gpu) and (not args.no_prefetch_to_gpu)

    # Train loop
    for epoch in range(start_epoch, args.epochs):
        if is_rank0:
            print(f"\n{'='*60}\nEpoch {epoch+1}/{args.epochs}\n{'='*60}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_accum=max(1, int(args.grad_accum)),
            prefetch_to_gpu=prefetch_to_gpu,
            max_grad_norm=float(args.max_grad_norm),
            log_interval=int(args.log_interval),
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                prefetch_to_gpu=prefetch_to_gpu,
            )
            if is_rank0:
                print(f"[Val] loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} top5={val_metrics['top5_accuracy']:.4f}")

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        if is_rank0:
            print(f"[Train] loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | lr={lr_now:.6g}")

        # wandb (rank0)
        if WANDB_AVAILABLE and (not args.no_wandb) and is_rank0:
            log = {
                "epoch": epoch + 1,
                "lr": lr_now,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
            }
            if val_metrics is not None:
                log.update({
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/top5_accuracy": val_metrics["top5_accuracy"],
                })
            wandb.log(log)

        # save ckpt (rank0 only)
        if is_rank0 and ((epoch + 1) % int(args.save_interval) == 0):
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            model_state = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model": model_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_val_acc": best_val_acc,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # best model (rank0 only)
        if is_rank0 and (val_metrics is not None) and (val_metrics["accuracy"] > best_val_acc):
            best_val_acc = val_metrics["accuracy"]
            best_path = os.path.join(args.output_dir, "best_model.pt")
            model_state = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
            torch.save(model_state, best_path)
            print(f"✓ New best! acc={best_val_acc:.4f} -> {best_path}")

        ddp_barrier()

    if is_rank0:
        final_path = os.path.join(args.output_dir, "final_model.pt")
        model_state = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
        torch.save(model_state, final_path)
        print(f"\nTraining complete. Saved final model to {final_path}")

    if WANDB_AVAILABLE and (not args.no_wandb) and is_rank0:
        wandb.finish()


if __name__ == "__main__":
    main()



