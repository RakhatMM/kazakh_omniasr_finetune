# Fine-tuning Kazakh ASR — step by step

Commands to fine-tune `omniASR_CTC_300M` on Kazakh and evaluate it. Run inside `tmux` so a dropped SSH
doesn't kill training.

## 1. Set up your session
```bash
export PY=/path/to/venv/bin/python                 # your python env
export REPO=/path/to/omnilingual-asr               # the cloned recipe repo
export HERE=$(pwd)                                  # this repo
export DATA=$HOME/kk_asr_data                       # where the parquet will go
export CKPT=$HOME/kk_asr_run                         # where checkpoints will go
export FAIRSEQ2_USER_ASSET_DIR=$HERE/cards           # fairseq2 reads your cards from here
export CUDA_VISIBLE_DEVICES=0                         # pick a free GPU (check `nvidia-smi`)
mkdir -p "$HERE/cards"
```

## 2. Get the data
- **KSC2** (training): download from HF `issai/Kazakh_Speech_Corpus_2` (10 split parts), then join + extract:
  ```bash
  cat ISSAI_KSC2.tar.gz.parta* | tar xz -C $HOME/ksc2          # -> $HOME/ksc2/.../ISSAI_KSC2/{Train,Test}
  export KSC2_DIR=$HOME/ksc2/ISSAI_KSC2
  ```
- **FLEURS** (training + eval): downloads automatically via `datasets` on first use — nothing to do.

## 3. Build the training parquet
```bash
$PY prep_data.py "$KSC2_DIR" "$DATA"
```
Produces `$DATA/version=0/...` (partitioned parquet) and `$DATA/language_distribution_0.tsv`.

## 4. Make the dataset card
The trainer gets the parquet path from a **card** (not the config). Create one with a unique name
(don't reuse `example_dataset` — it's a built-in and names must be unique):
```bash
cat > "$HERE/cards/kk_finetune.yaml" <<EOF
name: kk_finetune
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: $DATA/version=0
tokenizer_ref: omniASR_tokenizer_v1
EOF
```

## 5. Point the training config at your data
Copy a reference config and edit two things — the dataset name and the summary-tsv path:
```bash
cp configs/ctc-finetune.yaml my-train.yaml
#   in my-train.yaml set:
#     dataset.name: "kk_finetune"
#     mixture_parquet_storage_config.dataset_summary_path: "<$DATA>/language_distribution_0.tsv"
#   (training length, learning rate, batch/grad-accum, etc. are yours to choose)
```

## 6. Train
```bash
cd "$REPO"          # run from the repo root so `workflows.recipes...` imports
$PY -m workflows.recipes.wav2vec2.asr "$CKPT" --config-file "$HERE/my-train.yaml"
```
Checkpoints are written under `$CKPT/ws_*/checkpoints/step_*`. A FLEURS validation score is printed
periodically.

## 7. Make a model card for your trained checkpoint
```bash
cd "$HERE"
SDP=$(ls "$CKPT"/ws_*/checkpoints/step_*/model/pp_00/tp_00/sdp_00.pt | sort -V | tail -1)
cat > "$HERE/cards/my_ft.yaml" <<EOF
name: my_ft
model_family: wav2vec2_asr
model_arch: 300m
checkpoint: "$SDP"
tokenizer_ref: omniASR_tokenizer_v1
EOF
```

## 8. Evaluate
Compare the base model and your fine-tuned model. Check **Kazakh** (the target) **and other languages**
to see how fine-tuning affected them:
```bash
for M in omniASR_CTC_300M my_ft; do
  $PY eval.py $M kk_kz     # FLEURS Kazakh   (target)
  $PY eval.py $M en_us     # FLEURS English
  $PY eval.py $M ru_ru     # FLEURS Russian
  KSC2_DIR=$KSC2_DIR $PY eval.py $M ksc2     # KSC2 Kazakh test
done
```
Each prints WER/CER. Interpret the results yourself — across the target language and the others.
