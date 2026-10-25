#!/usr/bin/env python3
"""
train_anomaly.py — train RadarNet-1D on the RadioML-based anomaly dataset
==========================================================================
Same architecture and recipe as train.py (Adam 1e-3, wd 1e-4, cosine to 1e-6,
batch 256, early stopping patience 10), with the number of classes taken from
the dataset. Evaluation goes through anom_eval.report so it matches the
integer-model and baseline numbers exactly.

Usage:
    python train_anomaly.py --data-dir ../data_anom --out-dir ../checkpoints_anom
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from anom_eval import anomaly_score, load_meta, report
from model import RadarNet1D, count_parameters


def loader(data_dir, split, bs, shuffle):
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    ds = TensorDataset(torch.from_numpy(np.transpose(X, (0, 2, 1)).copy()),
                       torch.from_numpy(y.astype(np.int64)))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2,
                      pin_memory=torch.cuda.is_available()), y


@torch.no_grad()
def predict(model, dl, device):
    """Returns the logits (N, classes)."""
    model.eval()
    out = []
    for xb, _ in dl:
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("..", "data_anom"))
    ap.add_argument("--out-dir", default=os.path.join("..", "checkpoints_anom"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.data_dir, "dataset_info.json")) as f:
        classes = json.load(f)["classes"]
    tr, _ = loader(args.data_dir, "train", args.batch_size, True)
    va, y_va = loader(args.data_dir, "val", 1024, False)
    te, y_te = loader(args.data_dir, "test", 1024, False)

    model = RadarNet1D(in_channels=2, num_classes=len(classes)).to(device)
    print(f"RadarNet-1D: {count_parameters(model):,} parameters, {len(classes)} classes, {device}")
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    best, bad, hist = -1.0, 0, []
    best_path = os.path.join(args.out_dir, "radarnet_anom_best.pth")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        sched.step()
        va_acc = float((predict(model, va, device).argmax(1) == y_va).mean())
        hist.append(dict(epoch=ep, train_loss=tot / n, val_acc=va_acc))
        print(f"epoch {ep:3d}  loss {tot / n:.4f}  val acc {100 * va_acc:.2f}%  ({time.time() - t0:.0f}s)")
        if va_acc > best:
            best, bad = va_acc, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    logits = predict(model, te, device)
    np.save(os.path.join(args.out_dir, "cnn_fp32_test_logits.npy"), logits)
    report("cnn_fp32", y_te, logits.argmax(1), load_meta(args.data_dir, "test"), args.out_dir,
           classes, scores=anomaly_score(logits))
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(dict(history=hist, best_val_acc=best, params=count_parameters(model)), f, indent=2)
    print(f"best model → {best_path}")


if __name__ == "__main__":
    main()
