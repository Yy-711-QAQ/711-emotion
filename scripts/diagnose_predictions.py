import argparse
import pandas as pd
import numpy as np

PROB_COLS = [
    "prob_angry", "prob_disgusted", "prob_fearful",
    "prob_happy", "prob_neutral", "prob_sad", "prob_surprised"
]

def entropy(p):
    p = np.clip(p, 1e-8, 1.0)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    prob = df[PROB_COLS].to_numpy()

    top1 = prob.max(axis=1)
    sorted_prob = np.sort(prob, axis=1)
    margin = sorted_prob[:, -1] - sorted_prob[:, -2]
    ent = entropy(prob)

    print("==== Prediction diagnosis ====")
    print("num_samples:", len(df))
    print("mean prob per class:")
    for c, v in zip(PROB_COLS, prob.mean(axis=0)):
        print(f"  {c}: {v:.6f}")

    print("top1_prob:")
    print("  mean:", float(top1.mean()))
    print("  std :", float(top1.std()))
    print("  min :", float(top1.min()))
    print("  max :", float(top1.max()))

    print("margin(top1-top2):")
    print("  mean:", float(margin.mean()))
    print("  std :", float(margin.std()))
    print("  min :", float(margin.min()))
    print("  max :", float(margin.max()))

    print("binary entropy over 7 sigmoids:")
    print("  mean:", float(ent.mean()))
    print("  std :", float(ent.std()))

    pred_counts = df["pred_label"].value_counts().sort_index().to_dict()
    true_counts = df["true_label"].value_counts().sort_index().to_dict()
    print("true label counts:", true_counts)
    print("pred label counts:", pred_counts)

    # collapse heuristic
    collapsed = (
        top1.mean() < 0.62 and
        margin.mean() < 0.08 and
        prob.std() < 0.10
    )
    print("collapse_detected:", collapsed)

if __name__ == "__main__":
    main()
