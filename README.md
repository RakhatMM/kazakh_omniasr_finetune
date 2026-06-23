# Kazakh ASR fine-tuning

Fine-tune Meta's **Omnilingual ASR** CTC model (`omniASR_CTC_300M`) on Kazakh speech, and evaluate it
on FLEURS (Kazakh + other languages) and KSC2. Built around the official
[facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) training recipe.

See **[STEPS.md](STEPS.md)** for the exact commands.

## Data

| dataset | role | source |
|---|---|---|
| **KSC2** (Kazakh Speech Corpus 2) | training | HF: [`issai/Kazakh_Speech_Corpus_2`](https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2) |
| **FLEURS** | training + evaluation | HF: [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) (`kk_kz`; also `en_us`, `ru_ru` for eval) |
| **KSD** (Kazakh Speech Dataset) — *optional* | extra training data | OpenSLR: [SLR140](https://www.openslr.org/140/) (~554 h, CC BY-SA 3.0; 22–44 kHz → resample to 16 kHz mono) |

## Prerequisites
1. Clone the recipe repo: `git clone https://github.com/facebookresearch/omnilingual-asr` (the trainer lives here, not on PyPI).
2. A Python env with `omnilingual-asr`, `fairseq2`, `tensorboard`, `datasets`, `soundfile`, `jiwer`, `pyarrow`, `librosa`.
3. A GPU.

## Repo contents
```
prep_data.py    # convert audio + transcripts into the training parquet
eval.py         # score a model on FLEURS (kk/en/ru) or KSC2 test (WER/CER)
configs/        # reference YAMLs copied from the omnilingual-asr repo
  ctc-finetune.yaml
  ctc-finetune-recommendation.yaml
  example_dataset.yaml      # the dataset-card template
STEPS.md        # step-by-step commands
```

## Final dataset structure

The trainer reads an Apache **Parquet** dataset, **partitioned by `corpus` / `split` / `language`**
(Hive-style — the partition values live in the folder names, not inside the files).

**Per-row schema (one row = one utterance):**

| column | type | meaning |
|---|---|---|
| `text` | `string` | normalized transcript |
| `audio_bytes` | `list<int8>` | the audio file's **compressed** bytes (FLAC/OGG), 16 kHz mono |
| `audio_size` | `int64` | number of samples in the decoded waveform (duration = `audio_size / 16000`) |

`corpus`, `split`, `language` are **partition columns** (from the path), not stored per row.

**Directory layout:**
```
<dataset_root>/version=0/
├── corpus=ksc2/
│   ├── split=train/language=kaz_Cyrl/part-0.parquet …
│   └── split=dev/  language=kaz_Cyrl/part-0.parquet
└── corpus=fleurs/
    ├── split=train/language=kaz_Cyrl/part-0.parquet …
    └── split=dev/  language=kaz_Cyrl/part-0.parquet
<dataset_root>/language_distribution_0.tsv   # corpus | language | hours (for mixture weighting)
```

**How many pairs per parquet file?** As many as you like, but keep each file's `audio_bytes` column
**under ~2 GB** — a `list<int8>` uses 32-bit offsets (max ≈ 2.1 GB total bytes). With FLAC clips of
tens–hundreds of KB, that's a few thousand utterances per file; `prep_data.py` writes **~4000 rows per
part file** and uses `row_group_size=100` (for memory-friendly streaming and shuffling). Large datasets
simply become many `part-*.parquet` files across the partition folders.
