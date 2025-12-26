#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distill teacher policy/value into the student network.
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from model_pretrain import PretrainModel


def load_state_dict_compat(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


def _iter_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "*.npz")))


def _apply_action_mask(logits, mask):
    inf_mask = torch.clamp(torch.log(mask + 1e-45), min=-1e38, max=0)
    return logits + inf_mask


def distill(
    data_dir,
    device,
    hidden_dim,
    teacher_ckpt,
    student_ckpt_in,
    student_ckpt_out,
    epochs,
    batch_size,
    lr,
    hard_policy_weight,
    soft_policy_weight,
    value_weight,
    temperature,
):
    files = _iter_files(data_dir)
    if not files:
        raise FileNotFoundError("No npz files found in %s" % data_dir)

    sample = np.load(files[0])
    teacher_in_channels = sample["oracle_obs"].shape[1]

    teacher = PretrainModel(hidden_dim=hidden_dim, use_vec=False, in_channels=teacher_in_channels).to(device)
    load_state_dict_compat(teacher, teacher_ckpt)
    teacher.eval()

    student = PretrainModel(hidden_dim=hidden_dim, use_vec=True, in_channels=60).to(device)
    if student_ckpt_in:
        load_state_dict_compat(student, student_ckpt_in)
    student.train(True)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        total_count = 0
        for path in files:
            data = np.load(path)
            oracle_obs = data["oracle_obs"]
            student_obs = data["student_obs"]
            student_vec = data["student_vec"]
            student_mask = data["student_mask"]
            actions = data["action"].astype(np.int64)

            idx = np.random.permutation(len(oracle_obs))
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start : start + batch_size]
                o_obs = torch.tensor(oracle_obs[batch_idx], dtype=torch.float32, device=device)
                s_obs = torch.tensor(student_obs[batch_idx], dtype=torch.float32, device=device)
                s_vec = torch.tensor(student_vec[batch_idx], dtype=torch.float32, device=device)
                s_mask = torch.tensor(student_mask[batch_idx], dtype=torch.float32, device=device)
                act_t = torch.tensor(actions[batch_idx], dtype=torch.long, device=device)

                with torch.no_grad():
                    t_logits, t_values = teacher(o_obs)
                    t_logits = _apply_action_mask(t_logits, s_mask)

                s_logits, s_values = student(s_obs, s_vec, s_mask)

                loss = 0.0
                if hard_policy_weight > 0:
                    loss += hard_policy_weight * F.cross_entropy(s_logits, act_t)
                if soft_policy_weight > 0:
                    log_probs = F.log_softmax(s_logits / temperature, dim=1)
                    soft_targets = F.softmax(t_logits / temperature, dim=1)
                    loss += soft_policy_weight * F.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)
                if value_weight > 0:
                    loss += value_weight * F.mse_loss(s_values.squeeze(-1), t_values.squeeze(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)
                total_count += len(batch_idx)
        avg_loss = total_loss / max(total_count, 1)
        print("epoch %d loss %.6f" % (epoch + 1, avg_loss))

    if student_ckpt_out:
        torch.save(student.state_dict(), student_ckpt_out)
        print("saved:", student_ckpt_out)


def main():
    parser = argparse.ArgumentParser(description="Distill oracle teacher into student")
    parser.add_argument("--data_dir", required=True, help="Directory with oracle npz files")
    parser.add_argument("--teacher_ckpt", required=True, help="Teacher checkpoint")
    parser.add_argument("--student_ckpt_in", default="", help="Init student checkpoint (optional)")
    parser.add_argument("--student_ckpt_out", required=True, help="Output student checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Model hidden dim")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hard_policy_weight", type=float, default=0.5, help="CE weight on human actions")
    parser.add_argument("--soft_policy_weight", type=float, default=0.5, help="KL weight to teacher policy")
    parser.add_argument("--value_weight", type=float, default=0.1, help="MSE weight to teacher value")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    args = parser.parse_args()

    device = torch.device(args.device)
    distill(
        data_dir=args.data_dir,
        device=device,
        hidden_dim=args.hidden_dim,
        teacher_ckpt=args.teacher_ckpt,
        student_ckpt_in=args.student_ckpt_in,
        student_ckpt_out=args.student_ckpt_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hard_policy_weight=args.hard_policy_weight,
        soft_policy_weight=args.soft_policy_weight,
        value_weight=args.value_weight,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
