# Kazakh ASR fine-tuning

Fine-tune Meta's **Omnilingual ASR** CTC model (`omniASR_CTC_300M`) on Kazakh speech, and evaluate it
on FLEURS (Kazakh + other languages) and KSC2. Built around the official
[facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) training recipe.

See **[STEPS.md](STEPS.md)** for the pipeline explained stage by stage. You implement two pieces
yourself — a **data converter** (produces the parquet described under *Final dataset structure*) and an
**evaluation script** (WER/CER on FLEURS `kk`/`en`/`ru` + KSC2); STEPS.md is the spec for both.

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
STEPS.md        # the pipeline explained stage by stage (the spec for what you build)
configs/        # reference YAMLs from the omnilingual-asr repo (not yours to write)
  ctc-finetune.yaml
  ctc-finetune-recommendation.yaml
  example_dataset.yaml      # the dataset-card template
```
You write the rest yourself (see STEPS.md): a **data converter** → the parquet in *Final dataset
structure*, and an **evaluation script** (WER/CER on FLEURS kk/en/ru + KSC2). The training itself uses
the omnilingual-asr recipe + a config from `configs/`.

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
```
*(Optional: a `language_distribution_*.tsv` of `corpus | language | hours` is only needed if you enable
corpus/language **mixture weighting** via the `beta_*` settings. With a single language, set the betas to
`null` and skip it — that's what we do.)*

**How many pairs per parquet file?** As many as you like, but keep each file's `audio_bytes` column
**under ~2 GB** — a `list<int8>` uses 32-bit offsets (max ≈ 2.1 GB total bytes). With FLAC clips of
tens–hundreds of KB, that's a few thousand utterances per file — aim for **~4000 rows per part** and
`row_group_size=100` (for memory-friendly streaming and shuffling). Large datasets simply become many
`part-*.parquet` files across the partition folders.
