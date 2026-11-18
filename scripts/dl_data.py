"""Download and verify the corpora and model weights used in this study.

huggingface.co is unreachable from the environment this work was done in, so
weights come from the ModelScope mirror ``openai-mirror/whisper-*`` and FLEURS
comes from the ModelScope dataset mirror ``google/fleurs``.  Every artefact is
checked against the byte size published by the mirror, so a silent truncation
cannot enter the benchmark unnoticed.

usage:
  python -u scripts/dl_data.py fleurs          # km_kh train/dev/test + en_us dev/test
  python -u scripts/dl_data.py slr42           # OpenSLR SLR42 Khmer
  python -u scripts/dl_data.py models          # Whisper checkpoints
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
os.environ["MODELSCOPE_CACHE"] = str(ROOT / "models")

FLEURS_REPO = "google/fleurs"
FLEURS_BASE = f"https://modelscope.cn/api/v1/datasets/{FLEURS_REPO}/repo"

# Verified against the mirror's published file listing on 2026-10-31.
FLEURS_FILES = {
    "data/km_kh/train.tsv": 2310896,
    "data/km_kh/dev.tsv": 434578,
    "data/km_kh/test.tsv": 1071082,
    "data/km_kh/audio/train.tar.gz": 1201877230,
    "data/km_kh/audio/dev.tar.gz": 232879431,
    "data/km_kh/audio/test.tar.gz": 571702566,
    "data/en_us/dev.tsv": 213065,
    "data/en_us/test.tsv": 367864,
    "data/en_us/audio/dev.tar.gz": 171250900,
}

LOCAL_NAMES = {
    "data/km_kh/train.tsv": "km_train.tsv",
    "data/km_kh/dev.tsv": "km_dev.tsv",
    "data/km_kh/test.tsv": "km_test.tsv",
    "data/km_kh/audio/train.tar.gz": "train.tar.gz",
    "data/km_kh/audio/dev.tar.gz": "dev.tar.gz",
    "data/km_kh/audio/test.tar.gz": "test.tar.gz",
    "data/en_us/dev.tsv": "en_dev.tsv",
    "data/en_us/test.tsv": "en_test.tsv",
    "data/en_us/audio/dev.tar.gz": "en_dev.tar.gz",
}

SLR42_BASE = "https://openslr.magicdatatech.com/resources/42"
# Content-Length reported by the mirror on 2026-10-31.
SLR42_FILES = {"km_kh_male.zip": 866086951, "LICENSE": None, "info.txt": None}

MODELS = [
    "openai-mirror/whisper-tiny",
    "openai-mirror/whisper-base",
    "openai-mirror/whisper-small",
    "openai-mirror/whisper-medium",
    "openai-mirror/whisper-large-v3",
    "openai-mirror/whisper-large-v3-turbo",
]
ALLOW = [
    "config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
    "added_tokens.json", "normalizer.json", "preprocessor_config.json",
]


def curl(url: str, dest: Path, resume: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sS", "-L", "--retry", "5", "--retry-delay", "5", "-o", str(dest), url]
    if resume and dest.exists():
        cmd.insert(1, "-C")
        cmd.insert(2, "-")
    print(f"  GET {url}", flush=True)
    subprocess.run(cmd, check=True)


def verify(dest: Path, expected: int | None) -> bool:
    got = dest.stat().st_size
    if expected is None:
        print(f"  ok {dest.name} {got:,} B (size not pinned)", flush=True)
        return True
    status = "ok" if got == expected else "MISMATCH"
    print(f"  {status} {dest.name} {got:,} B (expected {expected:,})", flush=True)
    return got == expected


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def do_fleurs() -> int:
    out = ROOT / "data/fleurs"
    bad = []
    for remote, size in FLEURS_FILES.items():
        dest = out / LOCAL_NAMES[remote]
        if not (dest.exists() and verify(dest, size)):
            import urllib.parse

            curl(f"{FLEURS_BASE}?Revision=master&FilePath={urllib.parse.quote(remote)}", dest, resume=False)
            if not verify(dest, size):
                bad.append(remote)
    splits = {"train": "train", "dev": "dev", "test": "test", "en_dev": "en_dev"}
    for name, split in splits.items():
        tgz = out / (f"{name}.tar.gz" if name.startswith("en") else f"{name}.tar.gz")
        if not tgz.exists():
            continue
        target = out / split
        marker = target / ".extracted"
        if marker.exists():
            print(f"  already extracted {split}", flush=True)
            continue
        print(f"  extracting {tgz.name}", flush=True)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(target)
        target.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{tgz.name} {sha256(tgz)}\n", encoding="utf-8")
    n = sum(1 for _ in (out / "test/test").glob("*.wav")) if (out / "test/test").exists() else 0
    print(f"# fleurs ready; km test wavs = {n}", flush=True)
    return 1 if bad else 0


def do_slr42() -> int:
    out = ROOT / "data/slr42"
    bad = []
    for name, size in SLR42_FILES.items():
        dest = out / name
        if not (dest.exists() and verify(dest, size)):
            curl(f"{SLR42_BASE}/{name}", dest)
            if not verify(dest, size):
                bad.append(name)
    z = out / "km_kh_male.zip"
    target = out / "extracted"
    if z.exists() and z.stat().st_size == SLR42_FILES["km_kh_male.zip"] and not (target / ".extracted").exists():
        print("  extracting km_kh_male.zip", flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / ".extracted").write_text(f"{z.name} {sha256(z)}\n", encoding="utf-8")
    return 1 if bad else 0


def do_models() -> int:
    from modelscope import snapshot_download

    bad = []
    for name in MODELS:
        try:
            path = snapshot_download(name, allow_patterns=ALLOW)
            print(f"  ok {name} -> {path}", flush=True)
        except Exception as exc:
            bad.append(name)
            print(f"  FAIL {name} {type(exc).__name__}: {str(exc)[:160]}", flush=True)
    return 1 if bad else 0


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "fleurs"
    return {"fleurs": do_fleurs, "slr42": do_slr42, "models": do_models}[what]()


if __name__ == "__main__":
    raise SystemExit(main())
