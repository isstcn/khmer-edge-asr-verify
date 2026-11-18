"""Segmentation granularity of every Khmer reference source used in this benchmark.

Whisper's published FLEURS table reports Khmer as WER because Khmer is excluded from
the character-splitting applied to Chinese/Japanese/Thai/Lao/Burmese. That is only a
word error rate if the whitespace in the reference is word segmentation. This script
measures the granularity directly, in characters per whitespace token, against a
Khmer corpus that is genuinely word-segmented (OpenSLR SLR42) and against English
FLEURS as a same-pipeline control.

usage: segstats.py [--json results/segstats.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from khmerasr.corpus import hours, load_fleurs, load_slr42  # noqa: E402
from khmerasr.textnorm import ZWSP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    ("FLEURS km_kh train", "transcription", "data/fleurs/km_train.tsv", "data/fleurs/train/train", "km"),
    ("FLEURS km_kh dev", "transcription", "data/fleurs/km_dev.tsv", "data/fleurs/dev/dev", "km"),
    ("FLEURS km_kh test", "transcription", "data/fleurs/km_test.tsv", "data/fleurs/test/test", "km"),
    ("FLEURS en_us dev", "transcription", "data/fleurs/en_dev.tsv", "data/fleurs/en_dev/dev", "en"),
]


def stats(texts: list[str]) -> dict:
    toks = [t.split() for t in texts]
    ntok = sum(len(t) for t in toks)
    nchar = sum(len(t.replace(" ", "")) for t in texts)
    return {
        "n": len(texts),
        "tokens_per_utt": ntok / max(1, len(texts)),
        "chars_per_utt": nchar / max(1, len(texts)),
        "chars_per_token": nchar / max(1, ntok),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "results" / "segstats.json"))
    ap.add_argument("--slr42", default="data/slr42")
    args = ap.parse_args()

    res = {}

    for name, col, tsv, wavdir, _lang in SOURCES:
        p = ROOT / tsv
        if not p.exists():
            print(f"[skip] {name}", flush=True)
            continue
        us = load_fleurs(p, ROOT / wavdir)
        texts = [u.norm_ref for u in us]
        res[name] = {"column": col, "hours": round(hours(us), 3), **stats(texts)}
        # the same rows with the raw column and ZWSP promoted to a real separator
        raw = [u.raw_ref.replace(ZWSP, " ") for u in us]
        raw = [" ".join(t.split()) for t in raw]
        res[name + " [raw+ZWSP]"] = {
            "column": "raw_transcription, U+200B -> space",
            "hours": round(hours(us), 3),
            **stats(raw),
        }

    sl = load_slr42(ROOT / args.slr42)
    if sl:
        res["OpenSLR SLR42 km_kh_male"] = {
            "column": "line_index.tsv (word-segmented)",
            "hours": round(hours(sl), 3),
            **stats([u.norm_ref for u in sl]),
        }

    print(f"\n{'source':44s} {'n':>5s} {'hours':>7s} {'tok/utt':>8s} {'char/utt':>9s} "
          f"{'char/tok':>9s}", flush=True)
    print("-" * 90, flush=True)
    for k, v in res.items():
        print(f"{k:44s} {v['n']:5d} {v['hours']:7.3f} {v['tokens_per_utt']:8.2f} "
              f"{v['chars_per_utt']:9.1f} {v['chars_per_token']:9.2f}", flush=True)

    sl_key = "OpenSLR SLR42 km_kh_male"
    en_key = "FLEURS en_us dev"
    if sl_key in res:
        truth = res[sl_key]["chars_per_token"]
        print("\nratio to true Khmer word length (SLR42 = %.2f chars/token):" % truth, flush=True)
        for k, v in res.items():
            if k == sl_key:
                continue
            print(f"  {k:44s} {v['chars_per_token']/truth:5.2f}x", flush=True)
        res["_reference_chars_per_token"] = {
            "slr42_khmer_words": truth,
            "fleurs_en_words": res[en_key]["chars_per_token"] if en_key in res else None,
        }

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
