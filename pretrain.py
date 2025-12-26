#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监督学习预训练脚本 - 使用专家对局数据

训练目标：
1. Policy Loss: 交叉熵损失（预测专家动作）
2. Value Loss: MSE损失（预测折扣回报）
3. Total Loss: policy_loss + value_coeff * value_loss

集成wandb监控训练过程
"""

import os
import argparse
import json
from glob import glob
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed, logging disabled. Install: pip install wandb")

from model_pretrain import PretrainModel, LightPretrainModel


class MahjongDataset(Dataset):
    """麻将预训练数据集"""

    def __init__(self, data_dir, file_pattern="*.npz"):
        self.files = sorted(glob(os.path.join(data_dir, file_pattern)))
        if not self.files:
            raise ValueError(f"No data files found in {data_dir}")

        # 预加载所有数据（如果内存足够）
        print(f"Loading {len(self.files)} data files...")
        self.obs_list = []
        self.vec_list = []
        self.mask_list = []
        self.act_list = []
        self.value_list = []

        for file_path in tqdm(self.files, desc="Loading data"):
            data = np.load(file_path)
            self.obs_list.append(data['obs'])
            self.vec_list.append(data['vec'])
            self.mask_list.append(data['mask'])
            self.act_list.append(data['act'])
            self.value_list.append(data['value_target'])

        # 合并数据
        self.obs = np.concatenate(self.obs_list, axis=0)
        self.vec = np.concatenate(self.vec_list, axis=0)
        self.mask = np.concatenate(self.mask_list, axis=0)
        self.act = np.concatenate(self.act_list, axis=0)
        self.value = np.concatenate(self.value_list, axis=0)

        print(f"Total samples: {len(self.obs)}")
        print(f"  obs: {self.obs.shape}, {self.obs.dtype}")
        print(f"  vec: {self.vec.shape}, {self.vec.dtype}")
        print(f"  act: {self.act.shape}, {self.act.dtype}")
        print(f"  value: {self.value.shape}, {self.value.dtype}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            'obs': torch.from_numpy(self.obs[idx]).long(),
            'vec': torch.from_numpy(self.vec[idx].astype(np.float32)),
            'mask': torch.from_numpy(self.mask[idx]).long(),
            'act': torch.tensor(self.act[idx], dtype=torch.long),
            'value': torch.tensor(self.value[idx], dtype=torch.float32),
        }


def train_epoch(model, dataloader, optimizer, device, value_coeff=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training"):
        obs = batch['obs'].to(device)
        vec = batch['vec'].to(device)
        mask = batch['mask'].to(device)
        act = batch['act'].to(device)
        value_target = batch['value'].to(device)

        # 前向传播
        logits, value_pred = model(obs, vec, mask)

        # 策略损失（交叉熵）
        policy_loss = F.cross_entropy(logits, act)

        # 价值损失（MSE）
        value_loss = F.mse_loss(value_pred.squeeze(), value_target)

        # 总损失
        loss = policy_loss + value_coeff * value_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        total_loss += loss.item() * len(obs)
        total_policy_loss += policy_loss.item() * len(obs)
        total_value_loss += value_loss.item() * len(obs)

        # 计算准确率（top-1）
        pred_act = logits.argmax(dim=1)
        total_correct += (pred_act == act).sum().item()
        total_samples += len(obs)

    return {
        'loss': total_loss / total_samples,
        'policy_loss': total_policy_loss / total_samples,
        'value_loss': total_value_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }


@torch.no_grad()
def validate(model, dataloader, device, value_coeff=1.0):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_correct = 0
    total_top5_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Validating"):
        obs = batch['obs'].to(device)
        vec = batch['vec'].to(device)
        mask = batch['mask'].to(device)
        act = batch['act'].to(device)
        value_target = batch['value'].to(device)

        # 前向传播
        logits, value_pred = model(obs, vec, mask)

        # 损失
        policy_loss = F.cross_entropy(logits, act)
        value_loss = F.mse_loss(value_pred.squeeze(), value_target)
        loss = policy_loss + value_coeff * value_loss

        # 统计
        total_loss += loss.item() * len(obs)
        total_policy_loss += policy_loss.item() * len(obs)
        total_value_loss += value_loss.item() * len(obs)

        # Top-1准确率
        pred_act = logits.argmax(dim=1)
        total_correct += (pred_act == act).sum().item()

        # Top-5准确率
        top5_preds = logits.topk(5, dim=1)[1]
        total_top5_correct += sum((act[i] in top5_preds[i]) for i in range(len(act)))

        total_samples += len(obs)

    return {
        'loss': total_loss / total_samples,
        'policy_loss': total_policy_loss / total_samples,
        'value_loss': total_value_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'top5_accuracy': total_top5_correct / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Pretrain Mahjong model with expert data")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--val_data_dir", type=str, default=None, help="验证集目录（可选）")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="输出目录")
    parser.add_argument("--model_type", type=str, default="full", choices=["full", "light"], help="模型类型")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--use_vec", action="store_true", default=True, help="是否使用向量特征")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--value_coeff", type=float, default=1.0, help="价值损失系数")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")

    # Wandb参数
    parser.add_argument("--wandb_project", type=str, default="mahjong-pretrain", help="wandb项目名")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb运行名称")
    parser.add_argument("--no_wandb", action="store_true", help="禁用wandb")

    # 其他
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_interval", type=int, default=1, help="保存间隔（epoch）")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 初始化wandb
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
        )

    # 加载数据
    print("Loading training data...")
    train_dataset = MahjongDataset(args.data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )

    # 验证集（可选）
    val_loader = None
    if args.val_data_dir:
        print("Loading validation data...")
        val_dataset = MahjongDataset(args.val_data_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )

    # 创建模型
    print(f"Creating {args.model_type} model...")
    if args.model_type == "full":
        model = PretrainModel(hidden_dim=args.hidden_dim, use_vec=args.use_vec)
    else:
        model = LightPretrainModel(hidden_dim=args.hidden_dim)

    model = model.to(args.device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, args.value_coeff
        )

        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
              f"Policy: {train_metrics['policy_loss']:.4f} | "
              f"Value: {train_metrics['value_loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f}")

        # 验证
        if val_loader:
            val_metrics = validate(model, val_loader, args.device, args.value_coeff)
            print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
                  f"Policy: {val_metrics['policy_loss']:.4f} | "
                  f"Value: {val_metrics['value_loss']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"Top5: {val_metrics['top5_accuracy']:.4f}")

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Wandb日志
        if WANDB_AVAILABLE and not args.no_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train/loss": train_metrics['loss'],
                "train/policy_loss": train_metrics['policy_loss'],
                "train/value_loss": train_metrics['value_loss'],
                "train/accuracy": train_metrics['accuracy'],
            }
            if val_loader:
                log_dict.update({
                    "val/loss": val_metrics['loss'],
                    "val/policy_loss": val_metrics['policy_loss'],
                    "val/value_loss": val_metrics['value_loss'],
                    "val/accuracy": val_metrics['accuracy'],
                    "val/top5_accuracy": val_metrics['top5_accuracy'],
                })
            wandb.log(log_dict)

        # 保存checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics if val_loader else None,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # 保存最佳模型
        if val_loader and val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"✓ New best model! Val Acc: {best_val_acc:.4f}")

    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
