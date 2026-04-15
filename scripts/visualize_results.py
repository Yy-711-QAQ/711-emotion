import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report
)
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


def plot_confusion_matrix(y_true, y_pred, outdir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_roc_curves(y_true, y_prob, outdir):
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "roc_curves.png", dpi=300)
    plt.close()


def plot_pr_curves(y_true, y_prob, outdir):
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f"{cls} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "pr_curves.png", dpi=300)
    plt.close()


def plot_class_support(y_true, outdir):
    counts = pd.Series(y_true).value_counts().sort_index()
    values = [counts.get(i, 0) for i in range(len(CLASS_NAMES))]
    plt.figure(figsize=(8, 5))
    plt.bar(CLASS_NAMES, values)
    plt.xticks(rotation=20)
    plt.ylabel("Samples")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "class_distribution.png", dpi=300)
    plt.close()


def save_classification_report(y_true, y_pred, outdir):
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4
    )
    with open(outdir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="figures/results_vis")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    prob_cols = [
        "prob_angry", "prob_disgusted", "prob_fearful",
        "prob_happy", "prob_neutral", "prob_sad", "prob_surprised"
    ]
    y_prob = df[prob_cols].to_numpy()

    plot_confusion_matrix(y_true, y_pred, outdir)
    plot_roc_curves(y_true, y_prob, outdir)
    plot_pr_curves(y_true, y_prob, outdir)
    plot_class_support(y_true, outdir)
    save_classification_report(y_true, y_pred, outdir)

    print("Saved figures to:", outdir.resolve())
    for p in sorted(outdir.iterdir()):
        print("-", p.name)


if __name__ == "__main__":
    main()
