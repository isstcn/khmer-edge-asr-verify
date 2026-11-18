"""Unified ASR benchmark: accuracy and cost for Khmer speech recognition.

usage examples
  python -u scripts/bench.py --backend hf --model models/models/openai-mirror--whisper-small/snapshots/master \
      --device cuda --dtype float16 --decoding fallback --corpus fleurs_km_test --limit 40
  python -u scripts/bench.py --backend fw --model models/ct2/whisper-small-int8 --device cpu \
      --dtype int8 --decoding fallback --corpus fleurs_km_test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from khmerasr.corpus import load_fleurs, load_slr42  # noqa: E402
from khmerasr.metrics import BOOT_SEED, aggregate  # noqa: E402
from khmerasr.profiling import PowerPoller, peak_rss_mb  # noqa: E402
from khmerasr.textnorm import ALL_PROTOCOLS, PROTOCOL_BY_NAME  # noqa: E402

FALLBACK_TEMPS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

CORPORA = {
    "fleurs_km_train": ("data/fleurs/km_train.tsv", "data/fleurs/train/train", "km"),
    "fleurs_km_dev": ("data/fleurs/km_dev.tsv", "data/fleurs/dev/dev", "km"),
    "fleurs_km_test": ("data/fleurs/km_test.tsv", "data/fleurs/test/test", "km"),
    "fleurs_en_dev": ("data/fleurs/en_dev.tsv", "data/fleurs/en_dev/dev", "en"),
    "fleurs_en_test": ("data/fleurs/en_test.tsv", "data/fleurs/en_test/test", "en"),
    "slr42_km": ("data/slr42", None, "km"),
}


def load_corpus(name: str):
    tsv, wavdir, lang = CORPORA[name]
    if name == "slr42_km":
        return load_slr42(ROOT / tsv), lang
    return load_fleurs(ROOT / tsv, ROOT / wavdir), lang


# --------------------------------------------------------------------------- HF
class HFRunner:
    def __init__(self, model_path, device, dtype, decoding, adapter=None, nrn=0, rp=1.0, threads=0):
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.torch = torch
        dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        self.device = device
        self.decoding = decoding
        self.nrn = nrn
        self.rp = rp
        self.threads = threads or None
        t0 = time.perf_counter()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, dtype=dt)
        self.adapter = str(adapter) if adapter else None
        if self.adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter)
        self.model = self.model.to(device).eval()
        proc_src = self.adapter or model_path
        self.processor = AutoProcessor.from_pretrained(proc_src)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = None
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens = None
        self.load_s = time.perf_counter() - t0
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def transcribe(self, wav: Path, lang: str):
        import numpy as np
        import soundfile as sf
        import torch

        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.device, self.model.dtype)

        kw = dict(
            language=lang,
            task="transcribe",
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            max_new_tokens=440,
        )
        if self.decoding in ("beam5", "fallback"):
            kw["num_beams"] = 5
        else:
            kw["do_sample"] = False
        if self.decoding == "fallback":
            kw.update(
                temperature=FALLBACK_TEMPS,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_prev_tokens=False,
            )
        if self.nrn:
            kw["no_repeat_ngram_size"] = self.nrn
        if self.rp != 1.0:
            kw["repetition_penalty"] = self.rp
        with torch.inference_mode():
            out = self.model.generate(feats, **kw)
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        lp = self._confidence(out)
        return text, lp

    def _confidence(self, out):
        """Length-normalised mean log-probability of the returned sequence.

        For beam search transformers exposes ``sequences_scores`` directly; under
        greedy decoding it is absent and the score is recomputed from the
        per-step logits of the emitted tokens.
        """
        torch = self.torch
        ss = getattr(out, "sequences_scores", None)
        if ss is not None and len(ss):
            return float(ss[0])
        logits = getattr(out, "logits", None) or getattr(out, "scores", None)
        if not logits:
            return None
        # ``sequences`` carries Whisper's 4-token forced prefix but ``scores`` has
        # one entry per generated step, so align from the end rather than by index.
        toks = out.sequences[0, -len(logits):]
        vals = []
        for step, lg in enumerate(logits):
            row = lg[0] if lg.dim() == 2 else lg
            vals.append(float(torch.log_softmax(row.float(), -1)[toks[step]]))
        return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------- faster-whisper
class FWRunner:
    def __init__(self, model_path, device, dtype, decoding, adapter=None, nrn=0, rp=1.0, threads=0):
        from faster_whisper import WhisperModel

        if adapter:
            raise SystemExit("--adapter is only supported by the hf backend")
        self.decoding = decoding
        self.nrn = nrn
        self.rp = rp
        self.threads = threads or (16 if device == "cpu" else 4)
        t0 = time.perf_counter()
        self.model = WhisperModel(str(model_path), device=device, compute_type=dtype,
                                  cpu_threads=self.threads)
        self.load_s = time.perf_counter() - t0
        self.n_params = 0
        self.n_trainable = 0

    def transcribe(self, wav: Path, lang: str):
        kw = dict(language=lang, task="transcribe", condition_on_previous_text=False, vad_filter=False)
        if self.decoding in ("beam5", "fallback"):
            kw["beam_size"] = 5
        else:
            kw["beam_size"] = 1
        if self.decoding == "fallback":
            kw.update(
                temperature=list(FALLBACK_TEMPS),
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
            )
        else:
            kw["temperature"] = 0.0
        if self.nrn:
            kw["no_repeat_ngram_size"] = self.nrn
        if self.rp != 1.0:
            kw["repetition_penalty"] = self.rp
        segs, info = self.model.transcribe(str(wav), **kw)
        # transcribe() hands back a generator: iterating it for the text and then again for
        # the scores leaves the second pass empty, which is how every CT2 run of this
        # harness came to record a null confidence.
        segs = list(segs)
        text = "".join(s.text for s in segs).strip()
        lps = [s.avg_logprob for s in segs if s.avg_logprob is not None]
        return text, (sum(lps) / len(lps) if lps else None)


RUNNERS = {"hf": HFRunner, "fw": FWRunner}


def model_label(p: Path) -> str:
    """Human-readable model name; ModelScope mirrors store weights under .../snapshots/master."""
    if p.name in ("master", "main") and p.parent.name == "snapshots":
        repo = p.parents[2].name
        for prefix in ("openai-mirror--", "AI-ModelScope--", "models--"):
            repo = repo.replace(prefix, "")
        return repo
    return p.name


def _dir_bytes(p: Path) -> int | None:
    """Total size of a model directory, so a quantised build's footprint is recorded."""
    if not p.is_dir():
        return p.stat().st_size if p.exists() else None
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _mean_logprob(recs: list[dict]) -> float | None:
    """Mean per-utterance sequence score, ignoring utterances that yielded no segment.

    Recorded because the sequence score and the error rate can move in opposite
    directions under repetition suppression, which is the point of comparing them.
    """
    lps = [r["avg_logprob"] for r in recs if r.get("avg_logprob") is not None]
    return sum(lps) / len(lps) if lps else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(RUNNERS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="int8")
    ap.add_argument("--decoding", choices=["greedy", "beam5", "fallback"], default="fallback")
    ap.add_argument("--adapter", default="", help="path to a trained LoRA adapter directory (hf backend)")
    ap.add_argument("--nrn", type=int, default=0, help="no_repeat_ngram_size; 0 = off")
    ap.add_argument("--rp", type=float, default=1.0, help="repetition_penalty; 1.0 = off")
    ap.add_argument("--name", default="", help="model label used in outputs")
    ap.add_argument("--corpus", default="fleurs_km_test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all utterances")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--power", action="store_true", help="poll GPU board power")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for the fw backend; 0 keeps the existing default (16 on cpu, 4 on cuda)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    utts, lang = load_corpus(args.corpus)
    utts = utts[args.offset :]
    if args.stride > 1:
        utts = utts[:: args.stride]
    if args.limit:
        utts = utts[: args.limit]
    missing = [u for u in utts if not Path(u.wav).exists()]
    if missing:
        print(f"FATAL {len(missing)} missing wavs, e.g. {missing[0].wav}", flush=True)
        return 2

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    adapter = Path(args.adapter) if args.adapter else None
    if adapter is not None and not adapter.is_absolute():
        adapter = ROOT / adapter

    label = args.name or model_label(model_path)
    if adapter is not None:
        label = f"{label}+{adapter.parent.name}"

    runner = RUNNERS[args.backend](
        model_path, args.device, args.dtype, args.decoding, adapter=adapter, nrn=args.nrn, rp=args.rp,
        threads=args.threads,
    )
    print(
        f"# {args.backend} {label} device={args.device} dtype={args.dtype} "
        f"decoding={args.decoding} nrn={args.nrn} rp={args.rp} corpus={args.corpus} n={len(utts)} "
        f"threads={runner.threads} load={runner.load_s:.1f}s params={runner.n_params/1e6:.1f}M",
        flush=True,
    )

    poller = PowerPoller(interval=0.1)
    if args.power:
        poller.start()

    recs = []
    t_start = time.perf_counter()
    for i, u in enumerate(utts):
        t0 = time.perf_counter()
        hyp, lp = runner.transcribe(Path(u.wav), lang)
        wall = time.perf_counter() - t0
        recs.append(
            {
                "uid": u.uid, "duration": u.duration, "wall": wall, "rtf": wall / u.duration,
                "hyp": hyp, "avg_logprob": lp, "raw_ref": u.raw_ref, "norm_ref": u.norm_ref,
                "gender": u.gender, "rss_mb": peak_rss_mb(),
            }
        )
        if (i + 1) % 25 == 0 or i == len(utts) - 1:
            el = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(utts)}] {el:.0f}s elapsed, ETA {el/(i+1)*(len(utts)-i-1):.0f}s", flush=True)

    gpu = poller.stop() if args.power else {}
    total_wall = time.perf_counter() - t_start
    total_audio = sum(u.duration for u in utts)

    pairs_all = [(r["raw_ref"], r["hyp"]) for r in recs]
    pairs_norm = [(r["norm_ref"], r["hyp"]) for r in recs]
    proto_scores = {}
    for p in ALL_PROTOCOLS:
        if p.name.startswith("W1"):
            continue
        proto_scores[p.name] = aggregate(p, pairs_norm if p.name.startswith(("P2", "P3", "P4")) else pairs_all)
    primary = PROTOCOL_BY_NAME["P3_norm_nopunct"]
    cs = aggregate(primary, pairs_norm, boot=args.boot)

    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    extras = (f"_nrn{args.nrn}" if args.nrn else "") + (f"_rp{args.rp:g}" if args.rp != 1.0 else "")
    tag = args.tag or f"{args.backend}_{label}{extras}_{args.dtype}_{args.decoding}_{args.corpus}"

    summary = {
        "tag": tag, "backend": args.backend, "model": label, "model_path": str(model_path),
        "model_bytes": _dir_bytes(model_path),
        "adapter": str(adapter) if adapter else None,
        "device": args.device, "dtype": args.dtype, "decoding": args.decoding,
        "cpu_threads": runner.threads,
        "no_repeat_ngram_size": args.nrn, "repetition_penalty": args.rp,
        "corpus": args.corpus, "lang": lang, "n": len(utts),
        "model_load_s": runner.load_s, "n_params": runner.n_params,
        "n_trainable_params": runner.n_trainable,
        "total_audio_s": total_audio, "total_wall_s": total_wall,
        "rtf_corpus": total_wall / total_audio if total_audio else None,
        "rtf_mean_utt": sum(r["rtf"] for r in recs) / len(recs),
        "mean_avg_logprob": _mean_logprob(recs),
        "peak_rss_mb": max(r["rss_mb"] for r in recs),
        "gpu_power": gpu,
        "cer": cs.cer * 100, "cer_ci": [x * 100 for x in cs.cer_ci] if cs.cer_ci else None,
        "boot_resamples": args.boot, "boot_seed": BOOT_SEED,
        "ref_chars": cs.ref_chars, "hyp_chars": cs.hyp_chars,
        "mean_ref_chars": cs.ref_chars / cs.n if cs.n else None,
        "mean_hyp_chars": cs.hyp_chars / cs.n if cs.n else None,
        "frac_empty": cs.frac_empty, "frac_repetition": cs.frac_repetition,
        "frac_repetition6": cs.frac_repetition6, "mean_repetition6": cs.mean_repetition6,
        "frac_language_drift": cs.frac_language_drift, "mean_khmer_share": cs.mean_khmer_share,
        "frac_insertion_blowup": cs.frac_insertion_blowup,
        "sub": cs.substitutions, "ins": cs.insertions, "dele": cs.deletions,
        "protocol_cer": {k: v.cer * 100 for k, v in proto_scores.items()},
    }

    (outdir / f"{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    out_jsonl = args.out or str(outdir / f"{tag}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nSUMMARY " + json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"\nwrote {out_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
