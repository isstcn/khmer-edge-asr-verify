"""Download ASR model weights from ModelScope (huggingface.co is unreachable here).

Only the safetensors checkpoint is fetched; the mirrored repos also carry a
duplicate pytorch_model.bin and, for large-v3, extra fp32 shards that would
roughly triple the download for no benefit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MODELSCOPE_CACHE"] = str(ROOT / "models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from modelscope import snapshot_download  # noqa: E402

ALLOW = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "normalizer.json",
    "preprocessor_config.json",
    "sentencepiece.bpe.model",
]

# Ordered by importance to the study: the edge-plausible sizes first.
MODELS = [
    "openai-mirror/whisper-small",
    "openai-mirror/whisper-base",
    "openai-mirror/whisper-tiny",
    "openai-mirror/whisper-large-v3-turbo",
    "openai-mirror/whisper-medium",
    "openai-mirror/whisper-large-v3",
]


def main() -> int:
    only = sys.argv[1:] or MODELS
    failures = []
    for name in only:
        try:
            path = snapshot_download(name, allow_patterns=ALLOW)
            print(f"OK   {name} -> {path}", flush=True)
        except Exception as exc:  # keep going; one bad repo must not block the rest
            failures.append(name)
            print(f"FAIL {name} {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    print(f"\ndone. {len(only) - len(failures)}/{len(only)} ok", flush=True)
    if failures:
        print("failed:", ", ".join(failures), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
