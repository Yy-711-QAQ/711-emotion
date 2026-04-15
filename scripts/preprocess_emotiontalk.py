import os
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE = Path("data/EmotionTalk")
OUT_RAW = Path("data/EMOTIONTALK_RAW_PROCESSED")
OUT_SPLIT = Path("data/EMOTIONTALK_SPLIT")

MM_JSON_ROOT = BASE / "Multimodal" / "json"
MM_VIDEO_ROOT = BASE / "Multimodal" / "mp4"
TEXT_JSON_ROOT = BASE / "Text" / "json"
AUDIO_ROOT = BASE / "Audio" / "wav"

LABEL_MAP = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6,
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_split_from_utt_id(utt_id: str) -> str:
    group = utt_id.split("_")[0]
    if group in {"G01", "G12"}:
        return "valid"
    if group in {"G03", "G15"}:
        return "test"
    return "train"

def extract_frames(video_path: Path, out_dir: Path, fps: int = 5):
    ensure_dir(out_dir)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(out_dir / "image_%d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # ffmpeg 默认从 1 开始，改成 0 开始
    jpgs = sorted(out_dir.glob("image_*.jpg"))
    for i, p in enumerate(jpgs):
        p.rename(out_dir / f"tmp_{i}.jpg")
    for i, p in enumerate(sorted(out_dir.glob("tmp_*.jpg"))):
        p.rename(out_dir / f"image_{i}.jpg")

def one_hot_label(label_name: str):
    y = np.zeros(7, dtype=np.float32)
    y[LABEL_MAP[label_name]] = 1.0
    return y

def main():
    ensure_dir(OUT_RAW)
    ensure_dir(OUT_SPLIT)

    meta = {}
    train_ids, valid_ids, test_ids = [], [], []

    mm_json_files = sorted(MM_JSON_ROOT.rglob("*.json"))
    print(f"Found multimodal json files: {len(mm_json_files)}")

    kept = 0
    skipped = 0

    for mm_jpath in tqdm(mm_json_files):
        rel = mm_jpath.relative_to(MM_JSON_ROOT)           # G00001/.../xxx.json
        utt_id = mm_jpath.stem                            # G00001_12_02_009
        text_jpath = TEXT_JSON_ROOT / rel
        video_path = MM_VIDEO_ROOT / rel.with_suffix(".mp4")
        audio_path = AUDIO_ROOT / rel.with_suffix(".wav")

        if not text_jpath.exists():
            skipped += 1
            continue
        if not video_path.exists():
            skipped += 1
            continue
        if not audio_path.exists():
            skipped += 1
            continue

        try:
            with open(mm_jpath, "r", encoding="utf-8") as f:
                mm_obj = json.load(f)
            with open(text_jpath, "r", encoding="utf-8") as f:
                text_obj = json.load(f)
        except Exception:
            skipped += 1
            continue

        label_name = mm_obj.get("emotion_result", None)
        text = text_obj.get("content", None)

        if label_name not in LABEL_MAP:
            skipped += 1
            continue
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue

        out_dir = OUT_RAW / utt_id
        ensure_dir(out_dir)

        # 复制音频
        shutil.copy2(audio_path, out_dir / "audio.wav")

        # 抽帧
        try:
            extract_frames(video_path, out_dir, fps=5)
        except Exception:
            # 清理残留
            if (out_dir / "audio.wav").exists():
                os.remove(out_dir / "audio.wav")
            skipped += 1
            continue

        frame_files = list(out_dir.glob("image_*.jpg"))
        if len(frame_files) == 0:
            if (out_dir / "audio.wav").exists():
                os.remove(out_dir / "audio.wav")
            skipped += 1
            continue

        meta[utt_id] = {
            "text": text.strip(),
            "label": one_hot_label(label_name)
        }

        split = get_split_from_utt_id(utt_id)
        if split == "train":
            train_ids.append(utt_id)
        elif split == "valid":
            valid_ids.append(utt_id)
        else:
            test_ids.append(utt_id)

        kept += 1

    with open(OUT_RAW / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    for fname, ids in [
        ("train_split.txt", train_ids),
        ("valid_split.txt", valid_ids),
        ("test_split.txt", test_ids),
    ]:
        with open(OUT_SPLIT / fname, "w", encoding="utf-8") as f:
            for x in ids:
                f.write(x + "\n")

    print("\nDone.")
    print(f"kept   : {kept}")
    print(f"skipped: {skipped}")
    print(f"meta   : {len(meta)}")
    print(f"train/valid/test = {len(train_ids)}/{len(valid_ids)}/{len(test_ids)}")

if __name__ == "__main__":
    main()
    