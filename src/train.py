"""Train the pneumonia classifier.

Run from anywhere:
    python src/train.py --data-dir data/chest_xray --epochs 15

On Google Colab (free GPU): upload this repo, mount Drive or download the
dataset there, and run the same command.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/chest_xray")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = ChestXrayDataset(args.data_dir, "train")
    val_ds = ChestXrayDataset(args.data_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    # Counteract class imbalance so the minority class isn't ignored.
    labels = [label for _, label in train_ds.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"Train: {n_neg} normal, {n_pos} pneumonia "
          f"| pos_weight={pos_weight.item():.2f}")

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        val_loss, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
                all_labels.extend(y.cpu().numpy().ravel())
        val_loss /= len(val_ds)
        val_auc = roc_auc_score(all_labels, all_probs)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pth"))
            print(f"  -> saved new best (AUC {best_auc:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["val_auc"])
    axes[1].set_title("Validation AUC")
    fig.savefig(os.path.join(args.output_dir, "training_curves.png"))
    print(f"Done. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
