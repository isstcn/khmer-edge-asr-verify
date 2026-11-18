# Data and model card — Khmer edge-device ASR benchmark

All artifacts below were downloaded by `scripts/dl_data.py` and verified by byte size
against the serving host on the access date. Nothing is re-hosted; this card records
provenance so the benchmark can be reproduced exactly.

Access date for every row: **2026-10-31**.

---

## 1. Speech corpora

### 1.1 FLEURS (`google/fleurs`), splits `km_kh` and `en_us`

| Item | Value |
|---|---|
| Publisher | Google Research |
| Licence | CC BY 4.0 |
| Mirror used | `https://modelscope.cn/api/v1/datasets/google/fleurs/repo` |
| Reason for mirror | `huggingface.co` is not reachable from the build host; ModelScope mirrors the identical `google/fleurs` repository layout |
| Citation | Arora et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, SLT 2022 |

Pinned archive byte sizes (verified against `Content-Length` on the mirror):

| Remote path | Local path | Bytes |
|---|---|---|
| `data/km_kh/audio/dev.tar.gz` | `data/fleurs/dev.tar.gz` | 232,879,431 |
| `data/km_kh/audio/test.tar.gz` | `data/fleurs/test.tar.gz` | 571,702,566 |
| `data/km_kh/audio/train.tar.gz` | `data/fleurs/train.tar.gz` | 1,201,877,230 |
| `data/en_us/audio/dev.tar.gz` | `data/fleurs/en_dev.tar.gz` | 171,250,900 |
| `data/km_kh/dev.tsv` | `data/fleurs/km_dev.tsv` | 434,578 |
| `data/km_kh/test.tsv` | `data/fleurs/km_test.tsv` | 1,071,082 |
| `data/km_kh/train.tsv` | `data/fleurs/km_train.tsv` | 2,310,896 |
| `data/en_us/dev.tsv` | `data/fleurs/en_dev.tsv` | 213,065 |
| `data/en_us/test.tsv` | `data/fleurs/en_test.tsv` | 367,864 |

SHA-256 of the manifests actually consumed:

| File | SHA-256 |
|---|---|
| `km_train.tsv` | `88e78efbfeab505f65e66425c29e0551f6e8b7ce925d68f9976e05bff1a9d326` |
| `km_dev.tsv` | `68533bcb1204150931aa4625d4eaa113d5c2854ab66923d6b07f6bb067e198dc` |
| `km_test.tsv` | `bd6e8d46c3c9e19641dca57e77c8a4e430bbc8637e23895fdad5ac7a0b62e77e` |
| `en_dev.tsv` | `9d57ee7e91e9d4c92edb39f6bbea668ef8dc2a3ff96eb510d5580b2ad05d17ec` |
| `en_test.tsv` | `74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca` |

**Manifest schema (7 tab-separated columns).** Column indices are 0-based and are the
ones `khmerasr/corpus.py` reads.

| Index | Field | Used as |
|---|---|---|
| 0 | `id` | uid prefix |
| 1 | `filename` | wav path |
| 2 | `raw_transcription` | `raw_ref` |
| 3 | `transcription` | `norm_ref` |
| 4 | char-split transcription, `|`-delimited | segmentation analysis only |
| 5 | `num_samples` | duration = `num_samples / 16000` |
| 6 | `gender` | stratification |

Row counts and audio hours actually loaded (hours = `sum(num_samples) / 16000 / 3600`):

| Split | Rows | Audio (h) |
|---|---|---|
| `km_kh` train | 1,675 | 7.101 |
| `km_kh` dev | 326 | 1.290 |
| `km_kh` test | 771 | 3.151 |
| `en_us` dev | 394 | 1.046 |
| `en_us` test | 647 | 1.774 |

Adaptation trains on 1,578 utterances / 6.427 h of `km_kh` train and validates on
308 utterances / 1.177 h of `km_kh` dev; the remainder exceed Whisper's 444-token
decoder limit at 6 epochs and are dropped, not truncated.

### 1.2 OpenSLR SLR42 — *High quality TTS data for Khmer*

| Item | Value |
|---|---|
| Source | `https://openslr.magicdatatech.com/resources/42/km_kh_male.zip` |
| Licence | CC BY-SA 4.0 (verbatim from the shipped `LICENSE` / `info.txt`) |
| Description (verbatim `info.txt`) | "Multi-speaker TTS data for Khmer (km-KH) … file: km_kh_male.zip Khmer data from male speakers" |
| Pinned bytes | 866,086,951 (authoritative `Content-Length`) |
| Use in this work | Additional adaptation data only; never used for evaluation |

SLR42 is *read-aloud TTS prompt material*, i.e. a different genre from FLEURS. It is
therefore used exclusively to test whether adaptation scales with data volume, and is
held out of every reported error rate.

---

## 2. Model checkpoints

All six Whisper checkpoints come from the ModelScope `openai-mirror` organisation,
which re-hosts the OpenAI weights. Local layout is
`models/models/openai-mirror--<name>/snapshots/master/`.

**Provenance caveat, stated plainly.** The mirror exposes the revision as `master`
and does not publish a commit hash, so these checkpoints are pinned by byte size only.
Two independent checks are used instead of a hash:

1. `config.json` architecture fields match the published Whisper configurations
   (layer counts, `d_model`, `vocab_size`, `max_target_positions`, `num_mel_bins=128`
   for v3-family / 80 for v2-family).
2. An **English control**: each checkpoint is evaluated on FLEURS `en_us`. A corrupt or
   substituted download would degrade English too. English error rates are low for all
   six, so the Khmer collapse is a property of the models, not of the mirror.

| Checkpoint | `model.safetensors` bytes | Snapshot bytes | Params | enc/dec layers | `d_model` | mel bins | vocab |
|---|---|---|---|---|---|---|---|
| `whisper-tiny` | 151,061,672 | 155,434,424 | 37,760,640 | 4 / 4 | 384 | 80 | 51,865 |
| `whisper-base` | 290,403,936 | 294,776,748 | 72,593,920 | 6 / 6 | 512 | 80 | 51,865 |
| `whisper-small` | 966,995,080 | 971,367,937 | 241,734,912 | 12 / 12 | 768 | 80 | 51,865 |
| `whisper-medium` | 3,055,544,304 | 3,059,917,072 | 763,857,920 | 24 / 24 | 1024 | 80 | 51,865 |
| `whisper-large-v3` | 3,087,130,976 | 3,091,519,764 | 1,543,490,560 | 32 / 32 | 1280 | 128 | 51,866 |
| `whisper-large-v3-turbo` | 1,617,824,864 | 1,622,443,339 | 808,878,080 | 32 / 4 | 1280 | 128 | 51,866 |

Every architecture field above matches the published Whisper configuration, including
the two that distinguish the v3 generation: `num_mel_bins` 80 → 128 and `vocab_size`
51,865 → 51,866 (the added `yue` token). `large-v3-turbo` keeps the 32-layer encoder
and cuts the decoder to 4 layers. Parameter counts are computed from each checkpoint's
own `config.json` via `WhisperForConditionalGeneration(...).num_parameters()` on a meta
device, i.e. from the downloaded configs rather than from any published table.

### 2.1 Special-token drift between Whisper generations

`large-v3` and `large-v3-turbo` add Cantonese (`yue`) to the language list. `yue` is
appended at id 50358, immediately before the task tokens, so every task token shifts by
+1 while all language tokens at id ≤ 50357 — including `<|km|>` and `<|th|>` — are
unchanged. Hard-coding task-token ids across a family sweep is therefore a silent bug.
Values resolved from each checkpoint's own tokenizer:

| Model family | `<|startoftranscript|>` | `<|translate|>` | `<|transcribe|>` | `<|notimestamps|>` | `<|km|>` | `<|th|>` | `<|yue|>` |
|---|---|---|---|---|---|---|---|
| tiny / base / small / medium | 50258 | 50358 | 50359 | 50363 | 50323 | 50289 | absent (id 50257) |
| large-v3 / large-v3-turbo | 50258 | 50359 | 50360 | 50364 | 50323 | 50289 | 50358 |

All prefixes are resolved from the checkpoint's own tokenizer at run time, never
hard-coded; `scripts/diag_decoder.py` prints the resolved prefix for every run.

### 2.2 CTranslate2 conversions

Produced by `scripts/merge_export.py` from merged LoRA weights, under `models/ct2/`:
`whisper-small-float32` and `whisper-small-int8` exist for the unadapted baseline.
CTranslate2 reports `get_cuda_device_count() == 0` on this host, so all CT2 numbers are
CPU-only, at the supported compute types `float32 / int16 / int8 / int8_float32` — which is
the correct edge setting anyway.

---

## 3. Environment

| Item | Value |
|---|---|
| Interpreter | CPython 3.12 (Windows x64) |
| OS | Windows 10.0.26200, x64 |
| torch | 2.6.0+cu118 |
| CUDA | 11.8 |
| GPU | 11,264 MiB total |
| Seed | 20261031 (training and bootstrap) |

Bootstrap confidence intervals use 2,000 resamples over utterance indices with
`numpy.random.default_rng(20261031)`.

---

## 4. Reproducing

```bash
python scripts/dl_data.py                         # byte-verified corpus acquisition
python scripts/dl_models.py                       # checkpoint acquisition (mirror allow-list)
python scripts/refcols.py                         # reference-column analysis
python scripts/lora.py --help                     # matched-budget LoRA arms (see runs/*/meta.json)
python scripts/bench.py --backend hf --model <ckpt> --corpus fleurs_km_test \
    --decoding greedy --boot 2000 --tag <tag>
python scripts/merge_export.py --base whisper-small --quantization int8
```
