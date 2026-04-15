import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


CLASS_NAMES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
PROB_COLS = [
    "prob_angry", "prob_disgusted", "prob_fearful",
    "prob_happy", "prob_neutral", "prob_sad", "prob_surprised"
]


def plot_class_distribution(df, outdir):
    true_counts = df["true_label"].value_counts().sort_index()
    pred_counts = df["pred_label"].value_counts().sort_index()

    true_vals = [true_counts.get(i, 0) for i in range(len(CLASS_NAMES))]
    pred_vals = [pred_counts.get(i, 0) for i in range(len(CLASS_NAMES))]

    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, true_vals, width=width, label="True")
    plt.bar(x + width / 2, pred_vals, width=width, label="Pred")
    plt.xticks(x, CLASS_NAMES, rotation=20)
    plt.ylabel("Count")
    plt.title("True vs Predicted Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "true_vs_pred_distribution.png", dpi=300)
    plt.close()


def plot_probability_histograms(df, outdir, thresholds=None):
    for i, cls in enumerate(CLASS_NAMES):
        col = PROB_COLS[i]

        pos = df[df["true_label"] == i][col].values
        neg = df[df["true_label"] != i][col].values

        plt.figure(figsize=(8, 5))
        if len(neg) > 0:
            plt.hist(neg, bins=30, alpha=0.6, density=True, label="Negative")
        if len(pos) > 0:
            plt.hist(pos, bins=30, alpha=0.6, density=True, label="Positive")

        if thresholds is not None and i < len(thresholds):
            plt.axvline(thresholds[i], linestyle="--", linewidth=2, label=f"thr={thresholds[i]:.2f}")

        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title(f"Probability Distribution - {cls}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"prob_dist_{cls}.png", dpi=300)
        plt.close()


def save_uncertain_samples(df, outdir):
    prob = df[PROB_COLS].to_numpy()
    top1 = prob.max(axis=1)
    margin = np.sort(prob, axis=1)[:, -1] - np.sort(prob, axis=1)[:, -2]

    out = df.copy()
    out["top1_prob"] = top1
    out["margin_top1_top2"] = margin

    # 越不确定：top1 越低，margin 越小
    out = out.sort_values(["top1_prob", "margin_top1_top2"], ascending=[True, True])
    out.head(20).to_csv(outdir / "top20_most_uncertain_samples.csv", index=False)


def plot_confidence_boxplot(df, outdir):
    prob = df[PROB_COLS].to_numpy()
    top1 = prob.max(axis=1)
    tmp = df.copy()
    tmp["top1_prob"] = top1

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=tmp, x="true_label", y="top1_prob")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=20)
    plt.ylabel("Top-1 Probability")
    plt.xlabel("True Class")
    plt.title("Prediction Confidence by True Class")
    plt.tight_layout()
    plt.savefig(outdir / "confidence_by_true_class.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="figures/prediction_analysis")
    parser.add_argument("--thresholds", type=float, nargs="*", default=None,
                        help="Optional thresholds, e.g. 0.05 0.6 0.05 0.6 0.05 0.6 0.6")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    plot_class_distribution(df, outdir)
    plot_probability_histograms(df, outdir, thresholds=args.thresholds)
    plot_confidence_boxplot(df, outdir)
    save_uncertain_samples(df, outdir)

    print("Saved analysis files to:", outdir.resolve())
    for p in sorted(outdir.iterdir()):
        print("-", p.name)


if __name__ == "__main__":
    main()
