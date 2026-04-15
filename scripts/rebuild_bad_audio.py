import subprocess
from pathlib import Path

bad_list = Path("outputs/bad_audio_files.txt")
raw_root = Path("data/EmotionTalk/Multimodal/mp4")
proc_root = Path("data/EMOTIONTALK_RAW_PROCESSED")

if not bad_list.exists():
    raise SystemExit("outputs/bad_audio_files.txt not found")

for line in bad_list.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    bad_wav = line.split("\t")[0]
    bad_wav = Path(bad_wav)

    uid = bad_wav.parent.name  # e.g. G00003_11_02_002
    g1, g2, g3, g4 = uid.split("_", 3)
    rel_mp4 = Path(g1) / f"{g1}_{g2}" / f"{g1}_{g2}_{g3}" / f"{uid}.mp4"
    src_mp4 = raw_root / rel_mp4

    if not src_mp4.exists():
        print("missing mp4:", src_mp4)
        continue

    out_wav = proc_root / uid / "audio.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_mp4),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_wav)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("rebuilt:", out_wav)
    except subprocess.CalledProcessError:
        print("failed:", src_mp4)
