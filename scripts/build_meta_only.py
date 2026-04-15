import os
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE = Path("data/EmotionTalk")
TEXT_JSON = BASE / "Text/json"
MULTI_JSON = BASE / "Multimodal/json"
OUT_RAW = Path("data/EMOTIONTALK_RAW_PROCESSED")

LABEL_MAP = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6,
}

def main():
    meta = {}

    json_files = list(TEXT_JSON.rglob("*.json"))
    print(f"Found {len(json_files)} samples")

    for jf in tqdm(json_files):
        try:
            text_obj = json.load(open(jf, "r", encoding="utf-8"))
            content = text_obj.get("content", "").strip()

            if content == "":
                continue

            # 对应 Multimodal json
            rel_path = jf.relative_to(TEXT_JSON)
            multi_path = MULTI_JSON / rel_path

            if not multi_path.exists():
                continue

            multi_obj = json.load(open(multi_path, "r", encoding="utf-8"))
            emotion = multi_obj.get("emotion_result", "").lower()

            if emotion not in LABEL_MAP:
                continue

            # 构造 utt_id
            utt_id = jf.stem
            
            label = np.zeros(7, dtype=np.float32)
            label[LABEL_MAP[emotion]] = 1.0

            meta[utt_id] = {
                "text": content,
                "label": label,
            }

        except Exception as e:
            continue

    print("Saving meta.pkl ...")

    tmp_path = OUT_RAW / "meta.pkl.tmp"
    final_path = OUT_RAW / "meta.pkl"

    os.makedirs(OUT_RAW, exist_ok=True)

    with open(tmp_path, "wb") as f:
        pickle.dump(meta, f, protocol=4)

    os.replace(tmp_path, final_path)

    print("Done! meta size:", len(meta))


if __name__ == "__main__":
    main()
