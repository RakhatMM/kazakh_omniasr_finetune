# Fine-tuning Kazakh ASR — the pipeline (concepts)

This explains **what each stage does** and **the formats your code must produce/consume**. You write the
glue yourself (a data converter and an evaluation script) — the explanations below are the spec.

**Pipeline:** pretrained generalist (`omniASR_CTC_300M`) → fine-tune on Kazakh → evaluate.
Stages: get data → convert to the training format → register a dataset card → train → evaluate.

---

## 1. Get the data
- **KSC2** (training) — HF `issai/Kazakh_Speech_Corpus_2` (10 split parts → join + extract). Real-speech
  domains: crowdsourced / tv_news / radio / talkshow / podcasts / parliament (the `tts` domain is
  synthetic). Each utterance is a `.flac` + a matching `.txt` transcript.
- **FLEURS** (training + evaluation) — HF `google/fleurs` (`kk_kz`; plus `en_us`, `ru_ru` for eval),
  loaded via the `datasets` library.
- **Optional** — KSD, OpenSLR 140 (resample to 16 kHz mono first).
- **Split discipline:** fine-tune on *train*, hold out *dev/test*. Never train on what you'll evaluate.

## 2. Convert to the training format  *(write a converter)*
The trainer reads one **parquet dataset** in a specific shape. Your converter turns `(audio, transcript)`
pairs into it. Target format (full details in the README's *Final dataset structure*):
- partitioned by `corpus` / `split` / `language`;
- each row = `text` + `audio_bytes` (the audio **compressed** as FLAC, 16 kHz mono, stored as `list<int8>`)
  + `audio_size` (sample count);
- plus a `language_distribution_0.tsv` (corpus | language | hours).
> **Your job:** read KSC2 (`.flac`+`.txt`) and FLEURS, emit this parquet (+ tsv). Audio that's already
> 16 kHz mono FLAC can be copied byte-for-byte; anything else must be resampled/re-encoded first.

## 3. Register a dataset card
The trainer gets the parquet **path from a small asset card**, not from the training config. Create a card
(see `configs/example_dataset.yaml` for the shape) with a **unique** name and `data:` → your parquet's
`version=0` dir, and point the env var **`FAIRSEQ2_USER_ASSET_DIR`** at the folder holding your cards.
(Don't reuse built-in names like `example_dataset` — names must be unique or the trainer errors.)

## 4. Train  *(provided recipe — you invoke it, not write it)*
Use the omnilingual-asr training recipe with one of the configs in `configs/`. Run it **from the cloned
repo's root**, pointing at your config; it writes checkpoints + prints a periodic validation score.
The config exposes the **dials** you can experiment with:
- **learning rate** — the step size (too big overshoots / forgets; too small barely moves);
- **batch size + grad-accumulation** — how noisy vs. smooth each update's gradient is;
- **steps** — how long you train (too few = under-trained; too many = over-specialized).
Set `dataset.name` to your card and the `dataset_summary_path` to your tsv. *Choosing good values is part
of the exercise.*

## 5. Evaluate  *(write an eval script)*
Measure **WER / CER** (lower = better). Crucially, evaluate on **the target language *and* others**:
- FLEURS **Kazakh** (`kk_kz`) and **KSC2** test → did it get better at Kazakh?
- FLEURS **English** (`en_us`) and **Russian** (`ru_ru`) → did fine-tuning on Kazakh *hurt* other languages?
> **Your job:** load each test set, transcribe with the model (pass the right language code), compute
> WER/CER against the references. Compare the base model vs. your fine-tuned one across all four, and
> interpret what you see.
