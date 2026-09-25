"""Evaluate the trained model on the held-out test set.

    python src/evaluate.py --checkpoint results/best_model.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (RocCurveDisplay, classification_report,
                             confusion_matrix, roc_auc_score)
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, ChestXrayDataset
from model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/chest_xray")
    p.add_argument("--checkpoint", default="results/best_model.pth")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = ChestXrayDataset(args.data_dir, "test")
    loader = DataLoader(test_ds, batch_size=32)

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            labels.extend(y.numpy().ravel())
    probs = np.array(probs)
    labels = np.array(labels).astype(int)
    preds = (probs >= 0.5).astype(int)

    print(f"Test AUC-ROC: {roc_auc_score(labels, probs):.4f}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(labels, preds))

    RocCurveDisplay.from_predictions(labels, probs)
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"))
    print("Saved results/roc_curve.png")


if __name__ == "__main__":
    main()
