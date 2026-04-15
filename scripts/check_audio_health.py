import os
from pathlib import Path
import torchaudio

root = Path("data/EMOTIONTALK_RAW_PROCESSED")
bad = []
short = []
ok = 0

for wav in root.glob("*/audio.wav"):
    try:
        waveform, sr = torchaudio.load(str(wav))
        if waveform.numel() == 0:
            bad.append((str(wav), "empty"))
            continue
        dur = waveform.shape[1] / sr
        if dur < 0.3:
            short.append((str(wav), dur))
        ok += 1
    except Exception as e:
        bad.append((str(wav), repr(e)))

print("OK:", ok)
print("BAD:", len(bad))
print("SHORT(<0.3s):", len(short))

os.makedirs("outputs", exist_ok=True)

with open("outputs/bad_audio_files.txt", "w", encoding="utf-8") as f:
    for p, e in bad:
        f.write(f"{p}\t{e}\n")

with open("outputs/short_audio_files.txt", "w", encoding="utf-8") as f:
    for p, d in short:
        f.write(f"{p}\t{d:.4f}\n")

print("saved: outputs/bad_audio_files.txt")
print("saved: outputs/short_audio_files.txt")
