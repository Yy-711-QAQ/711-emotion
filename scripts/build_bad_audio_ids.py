from pathlib import Path

bad_txt = Path("outputs/bad_audio_files.txt")
out_txt = Path("outputs/bad_audio_ids.txt")

ids = []
for line in bad_txt.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    wav_path = line.split("\t")[0]
    sid = Path(wav_path).parent.name
    ids.append(sid)

ids = sorted(set(ids))
out_txt.write_text("\n".join(ids), encoding="utf-8")
print("saved:", out_txt)
print("num bad ids:", len(ids))
