import os
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd

# 统一成项目内部用的 7 类 id
LABEL2ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}

VALID_SPLITS = {"train", "valid", "test", "dev"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def normalize_split(split: str) -> str:
    s = split.strip().lower()
    if s == "dev":
        return "valid"
    if s not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown split: {split}")
    return s

def extract_frames(video_path: str, out_dir: Path, img_interval_ms: int = 500):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    frame_step = max(1, int(round(fps * img_interval_ms / 1000.0)))

    frame_idx = 0
    save_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            cv2.imwrite(str(out_dir / f"image_{save_idx}.jpg"), frame)
            save_idx += 1
        frame_idx += 1
    cap.release()

    if save_idx == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

def extract_audio(video_path: str, wav_path: Path):
    ensure_dir(wav_path.parent)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def build_processed(
    meta_csv: str,
    video_root: str,
    out_root: str,
    img_interval_ms: int = 500,
):
    """
    你需要准备一个 metadata csv，至少包含这些列：
    - sample_id
    - video_path   （相对 video_root 或绝对路径）
    - text
    - emotion
    - split        （train / valid 或 dev / test）

    例如：
    sample_id,video_path,text,emotion,split
    et_000001,session1/xxx.mp4,你好,happy,train
    """
    df = pd.read_csv(meta_csv)
    required_cols = {"sample_id", "video_path", "text", "emotion", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata csv 缺少列: {missing}")

    out_root = Path(out_root)
    raw_root = out_root / "EMOTIONTALK_RAW_PROCESSED"
    split_root = out_root / "EMOTIONTALK_SPLIT"
    ensure_dir(raw_root)
    ensure_dir(split_root)

    meta: Dict[str, Dict] = {}
    split_ids: Dict[str, List[str]] = {"train": [], "valid": [], "test": []}

    for row in df.itertuples(index=False):
        sample_id = str(row.sample_id)
        text = str(row.text)
        emotion = str(row.emotion).strip().lower()
        split = normalize_split(str(row.split))

        if emotion not in LABEL2ID:
            raise ValueError(f"Unknown emotion: {emotion}")

        vp = str(row.video_path)
        video_path = vp if os.path.isabs(vp) else os.path.join(video_root, vp)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")

        sample_dir = raw_root / sample_id
        ensure_dir(sample_dir)

        print(f"[INFO] processing {sample_id} ...")
        extract_frames(video_path, sample_dir, img_interval_ms=img_interval_ms)
        extract_audio(video_path, sample_dir / "audio.wav")

        meta[sample_id] = {
            "text": text,
            "label": LABEL2ID[emotion],   # 单标签 int，配合 CE
        }
        split_ids[split].append(sample_id)

    with open(raw_root / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    for split in ["train", "valid", "test"]:
        with open(split_root / f"{split}_split.txt", "w", encoding="utf-8") as f:
            for sid in split_ids[split]:
                f.write(sid + "\n")

    print("[DONE] Preprocessing finished.")
    print(f"[DONE] meta.pkl -> {raw_root / 'meta.pkl'}")
    print(f"[DONE] splits -> {split_root}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-csv", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, default="./data")
    parser.add_argument("--img-interval", type=int, default=500)
    args = parser.parse_args()

    build_processed(
        meta_csv=args.meta_csv,
        video_root=args.video_root,
        out_root=args.out_root,
        img_interval_ms=args.img_interval,
    )