"""Root-cause diagnostic for the Khmer transcription collapse.

Decodes one Khmer utterance under three forced prefixes -- ``transcribe/km``,
``translate/km`` and ``transcribe/th`` -- crossed with greedy, beam-5 and
temperature-fallback decoding, in both float16 and float32.

The ``transcribe/th`` cell is the control that does the work: the audio is
unchanged, only the language token in the forced prefix differs. If the acoustic
encoder were failing, the Thai cell would be as degenerate as the Khmer one.

Emits a human-readable log and a JSON record with, per cell, the resolved prefix
ids, whether EOS was emitted, the character n-gram repetition ratio, the Khmer
share of the output and CER against the reference -- so the manuscript can cite
these numbers from ``results/`` rather than transcribing them from a log.

usage: diag_decoder.py [--model openai-mirror--whisper-small ...] [--wav PATH]
                       [--out results/diag_decoder.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoProcessor, WhisperForConditionalGeneration  # noqa: E402

from khmerasr.metrics import (  # noqa: E402
    khmer_share,
    repetition_ratio,
    score_pair,
    script_profile,
)
from khmerasr.textnorm import PRIMARY  # noqa: E402

CELLS = (
    ("transcribe", "km"),
    ("translate", "km"),
    ("transcribe", "th"),
)
DECODINGS = (
    ("greedy", dict(do_sample=False, num_beams=1)),
    ("beam5", dict(num_beams=5)),
    ("fallback", dict(num_beams=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                      compression_ratio_threshold=2.4, logprob_threshold=-1.0,
                      no_speech_threshold=0.6, condition_on_prev_tokens=False)),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None,
                    help="repeatable; defaults to whisper-small and whisper-large-v3-turbo")
    ap.add_argument("--wav", default="data/fleurs/test/test/4555135160496574970.wav")
    ap.add_argument("--ref", default=None,
                    help="reference text; defaults to the km_test row matching --wav")
    ap.add_argument("--out", default=str(ROOT / "results" / "diag_decoder.json"))
    args = ap.parse_args()

    models = args.model or ["openai-mirror--whisper-small", "openai-mirror--whisper-large-v3-turbo"]
    wav = Path(args.wav)
    if not wav.is_absolute():
        wav = ROOT / wav

    ref = args.ref
    if ref is None:
        ref = _ref_for(wav.name)
    if not ref:
        raise SystemExit("could not resolve a reference; pass --ref explicitly")

    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    print(f"audio: {dur:.2f}s sr={sr} file={wav.name}")
    print(f"REF: {ref}")

    result = {"wav": str(wav), "duration_s": round(dur, 3), "sample_rate": sr,
              "ref": ref, "models": {}}

    for model_name in models:
        path = ROOT / "models/models" / model_name / "snapshots/master"
        proc = AutoProcessor.from_pretrained(path)
        tok = proc.tokenizer
        eos = tok.eos_token_id
        print(f"\ntokenizer vocab={len(tok)} km={tok.convert_tokens_to_ids('<|km|>')} "
              f"th={tok.convert_tokens_to_ids('<|th|>')} eos={eos}")
        mrec = {
            "vocab": len(tok),
            "feature_size": proc.feature_extractor.feature_size,
            "nb_max_frames": getattr(proc.feature_extractor, "nb_max_frames", None),
            "km_token_id": tok.convert_tokens_to_ids("<|km|>"),
            "th_token_id": tok.convert_tokens_to_ids("<|th|>"),
            "eos_token_id": eos,
            "dtypes": {},
        }
        result["models"][model_name] = mrec

        for dtype_name, dt in (("float16", torch.float16), ("float32", torch.float32)):
            print(f"\n{'='*70}\n### {model_name} {dtype_name}\n{'='*70}", flush=True)
            model = WhisperForConditionalGeneration.from_pretrained(path, dtype=dt).to("cuda").eval()
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = None
            model.generation_config.forced_decoder_ids = None
            model.generation_config.suppress_tokens = None
            feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dt)
            print(f"input_features shape={tuple(feats.shape)} "
                  f"min={feats.min().item():.3f} max={feats.max().item():.3f}")
            print(f"conv1 weight shape={tuple(model.model.encoder.conv1.weight.shape)}")

            drec = {
                "input_features_shape": list(feats.shape),
                "conv1_shape": list(model.model.encoder.conv1.weight.shape),
                "cells": {},
            }
            mrec["dtypes"][dtype_name] = drec

            for task, lang in CELLS:
                for decoding, extra in DECODINGS:
                    with torch.inference_mode():
                        out = model.generate(feats, language=lang, task=task,
                                             max_new_tokens=440,
                                             return_dict_in_generate=True,
                                             output_scores=True, **extra)
                    ids = out.sequences[0].tolist()
                    text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
                    ss = getattr(out, "sequences_scores", None)
                    score = float(ss[0]) if ss is not None and len(ss) else None
                    gen = ids[4:]                      # the 4 forced prefix tokens
                    sp = score_pair(PRIMARY(ref), PRIMARY(text), uid="")
                    prof = script_profile(text)
                    key = f"{task}/{lang}/{decoding}"
                    rec = {
                        "task": task, "lang": lang, "decoding": decoding,
                        "prefix_ids": ids[:4],
                        "prefix_tokens": [tok.decode([i]) for i in ids[:4]],
                        "ids_head": ids[:24],
                        "n_generated": len(gen),
                        "terminated_with_eos": bool(gen) and gen[-1] == eos,
                        "seq_score": score,
                        "text": text,
                        "cer": round(100.0 * sp.edits / max(1, sp.ref_chars), 2),
                        "repetition10": round(repetition_ratio(text, 10), 4),
                        "repetition6": round(repetition_ratio(text, 6), 4),
                        "khmer_share": round(khmer_share(text), 4),
                        "top_scripts": [f"{k}:{v}" for k, v in Counter(prof).most_common(3)],
                    }
                    drec["cells"][key] = rec
                    print(f"\n[{key}] ntok={rec['n_generated']} "
                          f"seq_score={'n/a' if score is None else f'{score:.2f}'} "
                          f"eos={rec['terminated_with_eos']}", flush=True)
                    print(f"  prefix={rec['prefix_tokens']}", flush=True)
                    print(f"  ids[:24]={rec['ids_head']}", flush=True)
                    print(f"  rep10={rec['repetition10']} rep6={rec['repetition6']} "
                          f"km_share={rec['khmer_share']} cer={rec['cer']}% "
                          f"scripts={rec['top_scripts']}", flush=True)
                    print(f"  OUT: {text[:220]}", flush=True)
            del model
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0


def _ref_for(wav_name: str) -> str:
    for tsv in ("data/fleurs/km_test.tsv", "data/fleurs/km_dev.tsv", "data/fleurs/km_train.tsv"):
        p = ROOT / tsv
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            f = line.rstrip("\n").split("\t")
            if len(f) >= 4 and f[1] == wav_name:
                return f[3]
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
