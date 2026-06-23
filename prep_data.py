"""Build the Omnilingual ASR training parquet from KSC2 (+ FLEURS-kk).

  train split = KSC2 real-domain Train (tts excluded) + FLEURS-kk train
  dev   split = FLEURS-kk test   (held-out eval used during training)

Also writes language_distribution_0.tsv (corpus | language | hours) for mixture weighting.

Usage:  python prep_data.py <KSC2_EXTRACTED_DIR> <OUT_DIR> [n_ksc2]
        KSC2_EXTRACTED_DIR must contain Train/<domain>/*.flac (+ matching *.txt)
"""
import io, os, sys, glob, random, csv
import numpy as np, soundfile as sf, pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset, Audio as HFAudio

KSC2_DIR = sys.argv[1]
OUT = sys.argv[2]
N_KSC2 = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
SR, LANG = 16000, "kaz_Cyrl"
REAL = ["crowdsourced", "tv_news", "radio", "talkshow", "podcasts", "parliament"]  # excludes synthetic 'tts'
root = f"{OUT}/version=0"
rng = random.Random(0)


def flac_bytes(w):                      # float32 mono @16k -> FLAC bytes as int8
    b = io.BytesIO(); sf.write(b, w, SR, format="FLAC"); return np.frombuffer(b.getvalue(), dtype=np.int8)


def write_chunks(rows, corpus, split, chunk=4000):
    d = f"{root}/corpus={corpus}/split={split}/language={LANG}"; os.makedirs(d, exist_ok=True)
    for i in range(0, len(rows), chunk):                        # keep each list<int8> array < 2 GB (int32 offsets)
        buf = rows[i:i + chunk]
        arrs = [r[1] for r in buf]
        off = np.zeros(len(arrs) + 1, dtype=np.int32); np.cumsum([len(a) for a in arrs], out=off[1:])
        ab = pa.ListArray.from_arrays(pa.array(off, pa.int32()), pa.array(np.concatenate(arrs), pa.int8()))
        pq.write_table(pa.table({"text": pa.array([r[0] for r in buf], pa.string()),
                                 "audio_bytes": ab,
                                 "audio_size": pa.array([r[2] for r in buf], pa.int64())}),
                       f"{d}/part-{i // chunk}.parquet", row_group_size=100)
    return len(rows), sum(r[2] for r in rows)


# --- KSC2 Train (16 kHz mono FLAC -> copy raw bytes) ---
flacs = []
for dom in REAL:
    flacs += glob.glob(f"{KSC2_DIR}/Train/{dom}/*.flac")
rng.shuffle(flacs)
ksc2 = []
for f in flacs:
    if len(ksc2) >= N_KSC2:
        break
    t = f[:-5] + ".txt"
    if not os.path.exists(t):
        continue
    txt = open(t, encoding="utf-8").read().strip()
    if not txt:
        continue
    info = sf.info(f)
    if info.samplerate != SR or info.channels != 1 or not (SR <= info.frames <= 39 * SR):
        continue
    ksc2.append((txt, np.frombuffer(open(f, "rb").read(), dtype=np.int8), info.frames))
n_k, s_k = write_chunks(ksc2, "ksc2", "train")


# --- FLEURS-kk (decoded arrays -> re-encode to FLAC) ---
def fleurs(split):
    ds = load_dataset("google/fleurs", "kk_kz", split=split).cast_column("audio", HFAudio(decode=False))
    rows = []
    for s in ds:
        w, _ = sf.read(io.BytesIO(s["audio"]["bytes"])); w = w.astype("float32")
        if SR <= len(w) <= 39 * SR:
            rows.append((s["transcription"], flac_bytes(w), len(w)))
    return rows


n_ftr, s_ftr = write_chunks(fleurs("train"), "fleurs", "train")
n_fdev, s_fdev = write_chunks(fleurs("test"), "fleurs", "dev")

# --- language_distribution_0.tsv (corpus | language | hours) ---
hours = {("ksc2", LANG): s_k, ("fleurs", LANG): s_ftr + s_fdev}
with open(f"{OUT}/language_distribution_0.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["corpus", "language", "hours"])
    for (corp, lng), samp in hours.items():
        w.writerow([corp, lng, f"{samp / 3600 / SR:.4f}"])

print(f"KSC2-train={n_k}  FLEURS-train={n_ftr}  FLEURS-dev(test)={n_fdev}")
print(f"parquet root: {root}")
print(f"summary tsv : {OUT}/language_distribution_0.tsv")
