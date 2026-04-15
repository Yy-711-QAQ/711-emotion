from pathlib import Path

ROOT = Path("data/EMOTIONTALK_RAW_PROCESSED")
OUT = Path("data/EMOTIONTALK_SPLIT")

def get_split(uid: str):
    group = uid.split("_")[0]
    if group in {"G00001", "G00012"}:
        return "valid"
    if group in {"G00003", "G00015"}:
        return "test"
    return "train"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_ids, valid_ids, test_ids = [], [], []

    ids = sorted([p.name for p in ROOT.iterdir() if p.is_dir()])

    for uid in ids:
        sp = get_split(uid)
        if sp == "train":
            train_ids.append(uid)
        elif sp == "valid":
            valid_ids.append(uid)
        else:
            test_ids.append(uid)

    for fname, arr in [
        ("train_split.txt", train_ids),
        ("valid_split.txt", valid_ids),
        ("test_split.txt", test_ids),
    ]:
        with open(OUT / fname, "w", encoding="utf-8") as f:
            for x in arr:
                f.write(x + "\n")

    print("train/valid/test =", len(train_ids), len(valid_ids), len(test_ids))

if __name__ == "__main__":
    main()

