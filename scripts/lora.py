"""Decoder-localised LoRA adaptation of Whisper for Khmer.

The cross-script probe in ``diag_decoder.py`` shows that Whisper's encoder
recovers Khmer content correctly -- forcing ``language="th"`` on Khmer audio
yields a fluent Thai transliteration whose content matches the reference --
while the Khmer decoder collapses into repetition loops.  This script tests
that localisation claim directly: encoder-only, decoder-only, and both arms are
trained at a *matched LoRA parameter budget* on the same in-domain Khmer
speech.  If the bottleneck really is the decoder prior, decoder-only adaptation
should match or beat the full model at a fraction of the parameters, and
encoder-only adaptation should buy almost nothing.

usage:
  python -u scripts/lora.py --base whisper-small --arm decoder --budget 4e6
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from khmerasr.corpus import load_fleurs, load_slr42  # noqa: E402

ARMS = {
    "encoder": r".*encoder\.layers\.\d+\.self_attn\.(q|v)_proj",
    "decoder": (
        r".*decoder\.layers\.\d+\.self_attn\.(q|v)_proj"
        r"|.*decoder\.layers\.\d+\.encoder_attn\.(q|v)_proj"
    ),
    "both": (
        r".*encoder\.layers\.\d+\.self_attn\.(q|v)_proj"
        r"|.*decoder\.layers\.\d+\.self_attn\.(q|v)_proj"
        r"|.*decoder\.layers\.\d+\.encoder_attn\.(q|v)_proj"
    ),
}


# --------------------------------------------------------------------- features
def build_cache(cache_dir: Path, split: str, utts, proc) -> list[tuple]:
    """Precompute mel features and reference token ids (no special tokens).

    ``cache_dir`` must be unique per training source: entries are named
    ``<split>_<index>.npy`` and existing files are reused without re-checking
    their content, so two different source lists sharing a directory would
    silently mix features with the wrong transcripts.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, u in enumerate(utts):
        stem = f"{split}_{i:05d}"
        fp, tp = cache_dir / f"{stem}.feat.npy", cache_dir / f"{stem}.tok.npy"
        if not (fp.exists() and tp.exists()):
            audio, sr = sf.read(str(u.wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            feat = proc(audio, sampling_rate=16000, return_tensors="np").input_features[0].astype(np.float16)
            np.save(fp, feat)
            ids = proc.tokenizer(u.norm_ref, add_special_tokens=False).input_ids
            np.save(tp, np.asarray(ids, dtype=np.int32))
        out.append((fp, tp, u))
        if (i + 1) % 500 == 0:
            print(f"  cached {split} {i+1}/{len(utts)}", flush=True)
    return out


# Whisper's decoder prompt is forced at inference, so training must see exactly
# the same prefix; otherwise the adapter is tuned under conditions the decoder
# never encounters at test time.
def decoder_prompt(tokenizer, lang: str, task: str = "transcribe") -> list[int]:
    toks = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>", "<|notimestamps|>"]
    ids = [tokenizer.convert_tokens_to_ids(t) for t in toks]
    if any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise ValueError(f"unresolved prompt tokens for {toks}: {ids}")
    return ids


def load_pairs(entries, max_text_tokens: int):
    feats, labels, durs, dropped = [], [], [], 0
    for fp, tp, u in entries:
        ids = np.load(tp)
        if len(ids) > max_text_tokens:
            dropped += 1
            continue
        feats.append(np.load(fp))
        labels.append(ids)
        durs.append(u.duration)
    return feats, labels, durs, dropped


def make_batch(feats, labels, idx, feature_size, nb_frames, device, dtype, prompt, eos_id):
    """Teacher-forced batch.

    With ``full = prompt + text + [eos]``, the decoder consumes ``full[:-1]`` and
    is trained to predict ``full[1:]``.  The three prompt positions whose targets
    are still prompt tokens are masked, because those tokens are forced at
    inference rather than predicted; the position that predicts the first text
    token and the one that predicts ``eos`` are both kept.
    """
    f = np.stack([feats[i] for i in idx]).astype(np.float32)
    if f.shape[1] < feature_size or f.shape[2] < nb_frames:
        pad = np.zeros((f.shape[0], feature_size, nb_frames), dtype=np.float32)
        pad[:, : f.shape[1], : f.shape[2]] = f
        f = pad
    np_prompt = np.asarray(prompt, dtype=np.int64)
    np_eos = np.int64(eos_id)
    dec = [np.concatenate([np_prompt, labels[i]]) for i in idx]
    lab = [np.concatenate([np.full(len(prompt) - 1, -100, dtype=np.int64), labels[i], [np_eos]]) for i in idx]
    dec_len = len(prompt) + max(len(labels[i]) for i in idx)
    dec_p = np.full((len(dec), dec_len), np_eos, dtype=np.int64)
    lab_p = np.full((len(lab), dec_len), -100, dtype=np.int64)
    for k, (d, l) in enumerate(zip(dec, lab)):
        dec_p[k, : len(d)] = d
        lab_p[k, : len(l)] = l
    return (
        torch.from_numpy(f).to(device, dtype),
        torch.from_numpy(dec_p).to(device, torch.long),
        torch.from_numpy(lab_p).to(device, torch.long),
    )


# --------------------------------------------------------------------- LoRA rank
def lora_params_per_rank(model, target_regex: str) -> tuple[int, int]:
    """Return (parameters added per rank unit, number of matched Linear layers)."""
    pat = re.compile(target_regex)
    per_rank, nmod = 0, 0
    for name, mod in model.named_modules():
        if pat.fullmatch(name) and hasattr(mod, "weight"):
            out_f, in_f = mod.weight.shape
            per_rank += int(out_f) + int(in_f)
            nmod += 1
    if per_rank == 0:
        raise RuntimeError(f"target regex matched no Linear layers: {target_regex}")
    return per_rank, nmod


def pick_rank(model, target_regex: str, budget: float) -> tuple[int, int, int]:
    per_rank, nmod = lora_params_per_rank(model, target_regex)
    return max(1, int(budget // per_rank)), per_rank, nmod


# ------------------------------------------------------------------------ train
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="whisper-small")
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--budget", type=float, default=4e6)
    ap.add_argument("--r", type=int, default=0, help="0 = auto-select to fit --budget")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lang", default="km")
    ap.add_argument("--train-tsv", default="data/fleurs/km_train.tsv")
    ap.add_argument("--train-wavdir", default="data/fleurs/train/train")
    ap.add_argument("--dev-tsv", default="data/fleurs/km_dev.tsv")
    ap.add_argument("--dev-wavdir", default="data/fleurs/dev/dev")
    ap.add_argument("--add-slr42", default="",
                    help="SLR42 root to append to the training source (adaptation only, "
                         "never evaluated on). Empty = FLEURS train alone.")
    ap.add_argument("--out", default="")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    base_path = ROOT / "models/models" / f"openai-mirror--{args.base}/snapshots/master"
    proc = AutoProcessor.from_pretrained(base_path)
    feature_size = proc.feature_extractor.feature_size
    nb_frames = proc.feature_extractor.nb_max_frames

    model = WhisperForConditionalGeneration.from_pretrained(base_path, dtype=dtype)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = None

    regex = ARMS[args.arm]
    if args.r:
        r = args.r
        per_rank, nmod = lora_params_per_rank(model, regex)
    else:
        r, per_rank, nmod = pick_rank(model, regex, args.budget)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=regex, bias="none")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"# base={args.base} arm={args.arm} modules={nmod} rank={r} "
        f"trainable={trainable:,} ({trainable / total * 100:.3f}% of {total:,}) device={device}",
        flush=True,
    )
    if args.report_only:
        print(json.dumps({"base": args.base, "arm": args.arm, "modules": nmod, "rank": r,
                          "params_per_rank": per_rank, "trainable": trainable, "total": total}, indent=2))
        return 0

    train_utts = load_fleurs(ROOT / args.train_tsv, ROOT / args.train_wavdir)
    src_name = "fleurs_km_train"
    if args.add_slr42:
        extra = load_slr42(ROOT / args.add_slr42)
        if not extra:
            raise SystemExit(f"--add-slr42 {args.add_slr42} yielded no utterances")
        # Held-out evaluation is always FLEURS dev, so appending read-aloud TTS
        # material to the training side cannot leak into any reported number.
        train_utts = train_utts + extra
        src_name = f"fleurs_km_train+slr42[{len(extra)}]"
        print(f"# train source = {src_name}", flush=True)

    # Dev keeps its own cache directory so the shared FLEURS dev features are not
    # recomputed for every training source.
    tr_cache = ROOT / "cache" / (args.base if not args.add_slr42 else f"{args.base}+slr42")
    dv_cache = ROOT / "cache" / args.base
    tr_entries = build_cache(tr_cache, "train", train_utts, proc)
    dv_entries = build_cache(dv_cache, "dev",
                             load_fleurs(ROOT / args.dev_tsv, ROOT / args.dev_wavdir), proc)
    prompt = decoder_prompt(proc.tokenizer, args.lang)
    eos_id = proc.tokenizer.eos_token_id
    max_text = model.config.max_target_positions - len(prompt)
    tr_f, tr_l, tr_d, tr_drop = load_pairs(tr_entries, max_text)
    dv_f, dv_l, dv_d, dv_drop = load_pairs(dv_entries, max_text)
    print(f"# prompt={prompt} eos={eos_id} max_text_tokens={max_text}", flush=True)
    print(f"# train={len(tr_f)}/{len(tr_entries)} (dropped {tr_drop}) dev={len(dv_f)}/{len(dv_entries)} "
          f"(dropped {dv_drop}) train_hours={sum(tr_d)/3600:.2f} dev_hours={sum(dv_d)/3600:.2f}", flush=True)

    model.to(device)
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(tr_f) / (args.batch * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warm_steps = max(1, int(args.warmup * total_steps))

    def lr_at(step: int) -> float:
        if step < warm_steps:
            return args.lr * step / warm_steps
        p = (step - warm_steps) / max(1, total_steps - warm_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    suffix = "_slr42" if args.add_slr42 else ""
    tag = args.out or f"lora_{args.base}_{args.arm}_r{r}_s{args.seed}{suffix}"
    outdir = ROOT / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def dev_loss() -> float:
        model.eval()
        tot, ntok = 0.0, 0
        for s in range(0, len(dv_f), args.batch):
            idx = list(range(s, min(s + args.batch, len(dv_f))))
            feats, dec, labels = make_batch(dv_f, dv_l, idx, feature_size, nb_frames, device, dtype, prompt, eos_id)
            out = model(input_features=feats, decoder_input_ids=dec, labels=labels)
            m = int((labels != -100).sum().item())
            tot += out.loss.item() * m
            ntok += m
        model.train()
        return tot / max(1, ntok)

    history = []
    best = (float("inf"), -1)
    step = 0
    t_start = time.perf_counter()
    for ep in range(args.epochs):
        order = list(range(len(tr_f)))
        random.shuffle(order)
        opt.zero_grad(set_to_none=True)
        run_loss, run_tok, t_ep = 0.0, 0, time.perf_counter()
        for bi in range(steps_per_epoch):
            idx = order[bi * args.batch * args.grad_accum : (bi + 1) * args.batch * args.grad_accum]
            for a in range(args.grad_accum):
                sub = idx[a * args.batch : (a + 1) * args.batch]
                if not sub:
                    continue
                feats, dec, labels = make_batch(tr_f, tr_l, sub, feature_size, nb_frames, device, dtype, prompt, eos_id)
                out = model(input_features=feats, decoder_input_ids=dec, labels=labels)
                (out.loss / args.grad_accum).backward()
                m = int((labels != -100).sum().item())
                run_loss += out.loss.item() * m
                run_tok += m
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep} step{step}/{total_steps} loss={run_loss/max(1,run_tok):.4f} "
                      f"lr={lr_at(step-1):.2e} {time.perf_counter()-t_ep:.0f}s", flush=True)
        dl = dev_loss()
        rec = {"epoch": ep, "train_loss": run_loss / max(1, run_tok), "dev_loss": dl,
               "epoch_seconds": round(time.perf_counter() - t_ep, 1), "steps": step}
        history.append(rec)
        print("# epoch " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if dl < best[0]:
            best = (dl, ep)
            model.save_pretrained(outdir / "adapter")
            proc.save_pretrained(outdir / "adapter")
            print(f"  -> saved best (dev_loss {dl:.4f})", flush=True)

    train_s = time.perf_counter() - t_start
    meta = {
        "base": args.base, "arm": args.arm, "rank": r, "lora_alpha": 2 * r,
        "lora_dropout": 0.05, "target_regex": regex, "modules_matched": nmod,
        "trainable_params": trainable, "total_params": total,
        "trainable_fraction": trainable / total, "budget": args.budget,
        "train_utterances": len(tr_f), "dev_utterances": len(dv_f),
        "train_hours": sum(tr_d) / 3600, "dev_hours": sum(dv_d) / 3600,
        "train_source": src_name, "train_tsv": args.train_tsv,
        "slr42_root": args.add_slr42 or None, "train_cache": str(tr_cache.relative_to(ROOT)),
        "epochs": args.epochs, "batch": args.batch, "grad_accum": args.grad_accum,
        "effective_batch": args.batch * args.grad_accum,
        "lr": args.lr, "scheduler": "cosine", "warmup_ratio": args.warmup,
        "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
        "seed": args.seed, "device": device, "dtype": "float32",
        "total_optimizer_steps": total_steps, "train_seconds": train_s,
        "best_dev_loss": best[0], "best_epoch": best[1], "history": history,
        "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"# done in {train_s/60:.1f} min; best dev_loss {best[0]:.4f} @ epoch {best[1]}\n# -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
