"""Merge a LoRA adapter into its base checkpoint and export CTranslate2 builds.

This is the deployment path used for the edge-cost study: the adapter is folded
into the base weights so the exported model has exactly the same architecture
and parameter count as the unadapted baseline, which makes the latency, memory
and energy comparisons attributable to quantization alone rather than to any
structural difference.

usage:
  # fold in a trained adapter, then export two quantizations
  python -u scripts/merge_export.py --adapter runs/lora_whisper-small_decoder_r54_s20261031/adapter \
      --base whisper-small --quantization float32 int8

  # no adapter: export the published checkpoint as-is, as the frontier's baseline
  python -u scripts/merge_export.py --base whisper-small \
      --quantization float32 float16 int8 int8_float16
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="",
                    help="LoRA adapter to fold in; empty converts the base checkpoint as-is")
    ap.add_argument("--base", default="whisper-small")
    ap.add_argument("--quantization", nargs="+", default=["float32", "int8"])
    args = ap.parse_args()

    base_path = ROOT / "models/models" / f"openai-mirror--{args.base}/snapshots/master"
    if not base_path.exists():
        base_path = Path(args.base)

    import ctranslate2

    t0 = time.perf_counter()
    if args.adapter:
        adapter = Path(args.adapter)
        if not adapter.is_absolute():
            adapter = ROOT / adapter
        from peft import PeftModel
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        model = WhisperForConditionalGeneration.from_pretrained(base_path, dtype=torch.float32)
        model = PeftModel.from_pretrained(model, str(adapter))
        merged = model.merge_and_unload()
        proc = AutoProcessor.from_pretrained(base_path)

        tag = adapter.parent.name
        merged_dir = ROOT / "models/merged" / tag
        merged_dir.parent.mkdir(parents=True, exist_ok=True)
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merged.save_pretrained(merged_dir, safe_serialization=True)
        proc.save_pretrained(merged_dir)
        n_params = sum(p.numel() for p in merged.parameters())
        print(f"# merged {n_params/1e6:.1f}M params in {time.perf_counter()-t0:.1f}s -> {merged_dir}",
              flush=True)
    else:
        # No adapter: convert the published checkpoint directly. The exported build
        # then differs from the adapted ones by quantization alone, which is the
        # comparison the cost frontier is making.
        adapter = None
        tag = args.base
        merged_dir = base_path
        from transformers import WhisperForConditionalGeneration

        n_params = sum(p.numel() for p in
                       WhisperForConditionalGeneration.from_pretrained(
                           base_path, dtype=torch.float32).parameters())
        print(f"# base {n_params/1e6:.1f}M params, converting in place", flush=True)

    merge_s = time.perf_counter() - t0

    builds = {}
    for q in args.quantization:
        out = ROOT / "models/ct2" / f"{tag}-{q}"
        if out.exists():
            shutil.rmtree(out)
        t1 = time.perf_counter()
        # This CT2 release has no thread knob on convert(), and copy_files belongs on the
        # converter, not the output call. faster-whisper loads tokenizer.json from the
        # model directory and would otherwise reach for the (unreachable) hub.
        ctranslate2.converters.TransformersConverter(
            str(merged_dir), copy_files=["tokenizer.json"]
        ).convert(str(out), quantization=q, force=True)
        tok_src = merged_dir / "tokenizer.json"
        tok_dst = out / "tokenizer.json"
        if tok_src.exists() and not tok_dst.exists():
            shutil.copy2(tok_src, tok_dst)
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        builds[q] = {"path": str(out), "bytes": size, "mb": round(size / 1e6, 1),
                     "convert_seconds": round(time.perf_counter() - t1, 1)}
        print(f"# ct2 {q}: {size/1e6:.1f} MB in {time.perf_counter()-t1:.1f}s -> {out}", flush=True)

    meta = {
        "adapter": str(adapter) if adapter else None, "base": str(base_path),
        "merged_dir": str(merged_dir), "tag": tag,
        "n_params": n_params, "merge_seconds": merge_s, "builds": builds,
    }
    outdir = ROOT / "models/merged"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{tag}.export.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
