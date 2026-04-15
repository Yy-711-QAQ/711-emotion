import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(text: str):
    data = {
        "train_losses": [],
        "valid_loss": None,
        "test_loss": None,
        "train_thresholds": None,
        "valid_thresholds": None,
        "test_avg_metrics": {},   # acc / recall / precision / f1 / auc
    }

    # 1) train loss
    train_loss_matches = re.findall(r"train loss:([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    data["train_losses"] = [float(x) for x in train_loss_matches]

    # 2) valid/test loss
    valid_match = re.search(r"valid loss:([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    test_match = re.search(r"test loss:([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)

    if valid_match:
        data["valid_loss"] = float(valid_match.group(1))
    if test_match:
        data["test_loss"] = float(test_match.group(1))

    # 3) thresholds
    train_thr_match = re.search(r"Train thresholds:\s*\[([^\]]+)\]", text)
    valid_thr_match = re.search(r"Valid thresholds:\s*\[([^\]]+)\]", text)

    if train_thr_match:
        data["train_thresholds"] = [float(x) for x in train_thr_match.group(1).split()]
    if valid_thr_match:
        data["valid_thresholds"] = [float(x) for x in valid_thr_match.group(1).split()]

    # 4) parse best performance tables
    # we look for the "average" of Test (1) in each metric block
    metric_names = ["acc", "recall", "precision", "f1", "auc"]
    for metric in metric_names:
        pattern = (
            rf"phase \({metric}\).*?"
            rf"Test \(1\)\s+([0-9\.\s]+)"
        )
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            nums = re.findall(r"[0-9]+\.[0-9]+", m.group(1))
            if nums:
                data["test_avg_metrics"][metric] = float(nums[-1])

    return data


def save_train_loss_curve(train_losses, outdir: Path):
    if not train_losses:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, linewidth=1.5)
    plt.xlabel("Logging Step")
    plt.ylabel("Train Loss")
    plt.title("Train Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "train_loss_curve.png", dpi=200)
    plt.close()


def save_phase_loss_bar(train_losses, valid_loss, test_loss, outdir: Path):
    if not train_losses and valid_loss is None and test_loss is None:
        return
    train_last = train_losses[-1] if train_losses else None

    labels, values = [], []
    if train_last is not None:
        labels.append("Train(last)")
        values.append(train_last)
    if valid_loss is not None:
        labels.append("Valid")
        values.append(valid_loss)
    if test_loss is not None:
        labels.append("Test")
        values.append(test_loss)

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Loss")
    plt.title("Phase Loss Comparison")
    plt.tight_layout()
    plt.savefig(outdir / "phase_loss_bar.png", dpi=200)
    plt.close()


def save_test_metrics_bar(test_avg_metrics, outdir: Path):
    if not test_avg_metrics:
        return
    labels = list(test_avg_metrics.keys())
    values = [test_avg_metrics[k] for k in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Test Average Metrics")
    plt.tight_layout()
    plt.savefig(outdir / "test_average_metrics_bar.png", dpi=200)
    plt.close()


def save_thresholds_bar(train_thr, valid_thr, outdir: Path):
    if train_thr is None and valid_thr is None:
        return

    class_names = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    x = range(len(class_names))

    plt.figure(figsize=(10, 5))
    width = 0.35

    if train_thr is not None:
        plt.bar([i - width / 2 for i in x], train_thr, width=width, label="Train")
    if valid_thr is not None:
        plt.bar([i + width / 2 for i in x], valid_thr, width=width, label="Valid")

    plt.xticks(list(x), class_names, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Threshold")
    plt.title("Per-class Thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "thresholds_bar.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to training log txt file")
    parser.add_argument("--outdir", type=str, default="figures/training_vis", help="Output directory")
    args = parser.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    data = parse_log(text)

    save_train_loss_curve(data["train_losses"], outdir)
    save_phase_loss_bar(data["train_losses"], data["valid_loss"], data["test_loss"], outdir)
    save_test_metrics_bar(data["test_avg_metrics"], outdir)
    save_thresholds_bar(data["train_thresholds"], data["valid_thresholds"], outdir)

    print("Saved figures to:", outdir.resolve())
    for p in sorted(outdir.glob("*.png")):
        print("-", p.name)


if __name__ == "__main__":
    main()
