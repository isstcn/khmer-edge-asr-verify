# Khmer edge-device ASR — verification code and measured evidence

This repository contains the evaluation code and the measured per-utterance evidence for a
benchmark of Whisper-family automatic speech recognition on Khmer (`google/fleurs`, split
`km_kh`). It is offered so that a reader can recompute the reported statistics and re-run the
pipeline. It does **not** contain the manuscript, its figures, or its bibliography; those travel
separately with the editorial submission and are not public here.

Everything under `results/` was produced by executing the code in this repository against the
pinned environment in `requirements.txt`. No statistic was entered by hand.

## Layout

```
khmerasr/            library
  corpus.py          FLEURS / SLR42 utterance loading
  textnorm.py        normalisation protocols P0-P4 and W1 (apply_protocol)
  metrics.py         score_pair, aggregate (micro-averaged CER over Unicode codepoints,
                     substitution/insertion/deletion split, script drift, repetition ratios,
                     percentile bootstrap), wer_space
  profiling.py       wall time, peak RSS, real-time factor, GPU board-power integral
scripts/
  bench.py           the single evaluation harness that produced every reported run
  refcols.py         FLEURS Khmer reference-column divergence and column x protocol CER grid
  segstats.py        characters-per-whitespace-token segmentation granularity across corpora
  diag_decoder.py    cross-script forced-decoding probe
  repsuppress.py     training-free repetition-suppression sweep
  lora.py            matched-parameter-budget LoRA training (manual peft + torch loop)
  merge_export.py    merge adapter -> CTranslate2 builds at the supported compute types
  dl_data.py         byte-verified corpus acquisition (sizes + SHA-256 checked)
  dl_models.py       checkpoint acquisition from a mirror, with an explicit file allow-list
  env_card.py        writes results/environment.json (machine, library versions, device count)
results/             60 *.json run summaries + 55 *.jsonl per-utterance sidecars
runs/                4 LoRA arms: meta.json (hyper-parameters and measured wall time) and
                     adapter/adapter_config.json (rank, alpha, target modules)
data/                DATA_CARD.md (provenance, licences, pinned byte sizes) and the SLR42
                     licence/info files
requirements.txt     the versions that actually produced results/, read back from the env
```

## Recompute offline — no GPU, no audio, no model download

Each `results/*.jsonl` sidecar carries, for every utterance, the reference text (`raw_ref`,
`norm_ref`) and the hypothesis text (`hyp`), plus timing and confidence fields. That is enough to
re-derive the corresponding `results/*.json` summary without re-running any model:

```python
import json, sys
sys.path.insert(0, ".")
from khmerasr.textnorm import P3
from khmerasr.metrics import aggregate

pairs = []
for line in open("results/edge_dec19_int8.jsonl", encoding="utf-8"):
    r = json.loads(line)
    pairs.append((r["norm_ref"], r["hyp"]))
score = aggregate(P3, pairs, boot=1000)   # bootstrap seed defaults to 20261031
print(score.cer)
```

`khmerasr/metrics.py` exposes `score_pair`, `aggregate` and `wer_space`; `khmerasr/textnorm.py`
defines the protocols `P0`-`P4` and `W1`. Feeding the `ref`/`hyp` pairs from a sidecar under each
protocol reproduces the matching `.json` summary in the same directory.

The reference-column and segmentation measurements need only the FLEURS split manifests, which
`scripts/dl_data.py` fetches into `data/fleurs/`:

```bash
python scripts/segstats.py
python scripts/refcols.py
```

## Re-run the pipeline

1. Create a Python 3.12 environment and install the pinned versions:

   ```bash
   pip install -r requirements.txt
   ```

   `torch`/`torchaudio` are pinned to the CUDA 11.8 wheels that produced these results; install
   the build appropriate for your own hardware if you differ. The CPU/CTranslate2 path needs no
   GPU.

2. Fetch the corpora and checkpoints (byte-size and SHA-256 verified; provenance in
   `data/DATA_CARD.md`):

   ```bash
   python scripts/dl_data.py
   python scripts/dl_models.py
   ```

   The build host could not reach `huggingface.co`, so `dl_models.py` pulls the published OpenAI
   Whisper checkpoints from ModelScope mirrors (`openai-mirror/whisper-*`) restricted to an
   explicit `ALLOW` file list. A reader with normal egress can point the same script at the
   upstream hosts. Both download scripts resolve paths relative to the repository root, so no
   absolute path needs editing.

3. Record the environment, then evaluate. `bench.py` is the harness for every run; see
   `python scripts/bench.py --help` for all flags. Representative cells:

   ```bash
   python scripts/env_card.py

   # CPU int8 edge cell (faster-whisper / CTranslate2 backend)
   python scripts/bench.py --backend fw --device cpu --dtype int8 --threads 4 \
       --corpus fleurs_km_test --stride 4 --boot 1000 --tag edge_base_int8 \
       --model models/ct2/whisper-small-int8 --decoding greedy --nrn 3 --rp 1.2

   # HF backend on GPU, unadapted
   python scripts/bench.py --backend hf --device cuda --dtype float16 \
       --corpus fleurs_km_test --stride 4 --boot 2000 --tag kmtest_base \
       --model models/models/openai-mirror--whisper-small/snapshots/master
   ```

   LoRA arms are trained with `scripts/lora.py` (hyper-parameters in each `runs/*/meta.json`) and
   exported to CTranslate2 with `scripts/merge_export.py`. The trained adapter weights are not
   redistributed here; `meta.json` plus `adapter/adapter_config.json` give everything needed to
   retrain each arm.

## Hardware and environment

`results/environment.json` records the machine, library versions and device count. On this host
`ctranslate2` reported zero CUDA devices, so the on-device cost measurements are CPU-only by
measurement. The supported CPU compute types are read from that card rather than assumed.

## What is not here, and why

| Omitted | Reason |
|---|---|
| Manuscript, figures, bibliography | Not public; they travel with the editorial submission |
| `models/` checkpoints (~14 GB) | Published OpenAI Whisper weights; `scripts/dl_models.py` fetches them |
| `data/fleurs/*.tar.gz`, `data/slr42/*.zip` audio (~7 GB) | Redistributed by the original publishers; `scripts/dl_data.py` pins byte sizes and SHA-256 |
| `runs/*/adapter/adapter_model.safetensors` | Derived fine-tuned weights, retrainable from `meta.json` + `adapter_config.json`; the SLR42 arm also inherits a ShareAlike corpus |
| Derived caches, machine-specific shell drivers | Not needed to recompute; the `.py` commands above replace the drivers |

## Licence

The code is MIT-licensed (`LICENSE`). The corpora and checkpoints carry their own licences and
are not covered by the MIT grant; see `NOTICE` and `data/DATA_CARD.md`. FLEURS reference text is
embedded in the sidecars under CC BY 4.0; SLR42 is CC BY-SA 4.0 and is used only for training,
not redistributed as audio here.