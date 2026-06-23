"""Score a model card on FLEURS (kk/en/ru) or KSC2 test, with jiwer (WER/CER).

Usage:  python eval.py <model_card> <dataset>
  <dataset> = kk_kz | en_us | ru_ru   (FLEURS test for that language)
            | ksc2                     (KSC2 Kazakh test; needs KSC2_DIR env var)

The model card must be discoverable (e.g. set FAIRSEQ2_USER_ASSET_DIR to your cards/ folder).
Evaluating the target language AND others shows how fine-tuning affected each.
"""
import io, os, re, sys, glob, random
import soundfile as sf
from datasets import load_dataset, Audio as HFAudio
from jiwer import wer, cer
from transformers import logging as L; L.set_verbosity_error()
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

card, which = sys.argv[1], sys.argv[2]
SR = 16000
FLEURS2OMNI = {"kk_kz": "kaz_Cyrl", "en_us": "eng_Latn", "ru_ru": "rus_Cyrl"}
norm = lambda t: re.sub(r"[^\w\s]", " ", t.lower()).strip()

clips, refs = [], []
if which in FLEURS2OMNI:
    lang = FLEURS2OMNI[which]
    ds = load_dataset("google/fleurs", which, split="test").cast_column("audio", HFAudio(decode=False))
    for s in ds:
        w, _ = sf.read(io.BytesIO(s["audio"]["bytes"])); w = w.astype("float32")
        if SR <= len(w) <= 39 * SR:
            clips.append(w); refs.append(s["transcription"])
elif which == "ksc2":
    lang = "kaz_Cyrl"
    base = os.environ["KSC2_DIR"] + "/Test"     # set KSC2_DIR to the extracted ISSAI_KSC2 dir
    fl = []
    for d in ["crowdsourced", "tv_news", "radio", "talkshow", "podcasts", "parliament"]:
        fl += glob.glob(f"{base}/{d}/*.flac")
    random.Random(0).shuffle(fl)
    for f in fl:
        if len(clips) >= 500:
            break
        t = f[:-5] + ".txt"
        if not os.path.exists(t):
            continue
        txt = open(t, encoding="utf-8").read().strip()
        if not txt:
            continue
        w, _ = sf.read(f)
        if getattr(w, "ndim", 1) > 1:
            w = w.mean(1)
        clips.append(w.astype("float32")); refs.append(txt)
else:
    sys.exit(f"unknown dataset '{which}' (use kk_kz | en_us | ru_ru | ksc2)")

os.makedirs("/tmp/asr_eval", exist_ok=True)
files = [f"/tmp/asr_eval/{i}.wav" for i in range(len(clips))]
for p, a in zip(files, clips):
    sf.write(p, a, SR)

pipe = ASRInferencePipeline(model_card=card)
hyp = [str(h) for h in pipe.transcribe(files, lang=[lang] * len(files), batch_size=16)]
W = sum(wer(norm(r), norm(x)) for r, x in zip(refs, hyp)) / len(refs)
C = sum(cer(norm(r), norm(x)) for r, x in zip(refs, hyp)) / len(refs)
print(f"{card} on {which}: WER={W:.3f}  CER={C:.3f}  (n={len(files)})")
