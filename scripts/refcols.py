"""Characterise the two FLEURS Khmer reference columns and the effect on reported CER.

usage: refcols.py [--json results/refcols.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from khmerasr.metrics import edit_distance  # noqa: E402
from khmerasr.textnorm import ALL_PROTOCOLS, PRIMARY, W1, ZWSP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "km_train": "data/fleurs/km_train.tsv",
    "km_dev": "data/fleurs/km_dev.tsv",
    "km_test": "data/fleurs/km_test.tsv",
    "en_dev": "data/fleurs/en_dev.tsv",
    "en_test": "data/fleurs/en_test.tsv",
}


def cat_hist(text: str) -> Counter:
    return Counter(unicodedata.category(c) for c in text)


def _merge_cats(texts: list[str]) -> Counter:
    tot = Counter()
    for t in texts:
        tot.update(cat_hist(t))
    return tot


def _mean_edit_sample(pairs: list[tuple[str, str]], cap: int) -> float:
    step = max(1, len(pairs) // cap)
    sub = pairs[::step][:cap]
    return sum(edit_distance(a, b) for a, b in sub) / max(1, len(sub))


def _agree(tok_norm: list[list[str]], split: list[str]) -> float:
    """Rows whose whitespace tokens equal the char-split column's |-segments."""
    k = 0
    for toks, sp in zip(tok_norm, split):
        segs = [s for s in ("".join(x.split()) for x in sp.split("|")) if s]
        if segs == toks:
            k += 1
    return k / max(1, len(tok_norm))


def analyse(path: Path) -> dict:
    rows = []
    ncols = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.rstrip("\n").split("\t")
        ncols[len(p)] += 1
        if len(p) < 7:
            continue
        rows.append(p)

    n = len(rows)
    raw, norm, split = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows]

    out = {
        "n": n,
        "ncols": dict(ncols),
        "frac_raw_eq_norm": sum(a == b for a, b in zip(raw, norm)) / max(1, n),
        "zwsp_rows_raw": sum(ZWSP in t for t in raw),
        "zwsp_rows_norm": sum(ZWSP in t for t in norm),
        "zwsp_rows_split": sum(ZWSP in t for t in split),
        "mean_zwsp_raw": sum(t.count(ZWSP) for t in raw) / max(1, n),
        "mean_len_raw": sum(len(t) for t in raw) / max(1, n),
        "mean_len_norm": sum(len(t) for t in norm) / max(1, n),
        "cats_raw": dict(_merge_cats(raw)),
    }

    # Where the two columns differ, how do they differ?
    diffs = [(a, b) for a, b in zip(raw, norm) if a != b]
    if diffs:
        out["n_diff"] = len(diffs)
        out["frac_diff_only_punct"] = sum(
            1 for a, b in diffs
            if "".join(c for c in a if unicodedata.category(c)[0] not in "PC")
            == "".join(c for c in b if unicodedata.category(c)[0] not in "PC")
        ) / len(diffs)
        out["frac_diff_only_trailing"] = sum(
            1 for a, b in diffs if a.startswith(b) or b.startswith(a)
        ) / len(diffs)
        out["mean_editdist_diff"] = _mean_edit_sample(diffs, 400)
        # characters present in raw but not norm, aggregated
        extra = Counter()
        for a, b in diffs:
            extra.update(set(a) - set(b))
        out["chars_only_in_raw"] = {
            c: {"n": k, "name": _name(c)} for c, k in extra.most_common(12)
        }
        missing = Counter()
        for a, b in diffs:
            missing.update(set(b) - set(a))
        out["chars_only_in_norm"] = {
            c: {"n": k, "name": _name(c)} for c, k in missing.most_common(12)
        }

    # The char-split column: does it reconstruct the transcription?
    recon = [t.replace(" ", "").replace("|", " ").strip() for t in split]
    out["frac_split_reconstructs_norm"] = sum(
        1 for a, b in zip(recon, norm) if a.replace(" ", "") == b.replace(" ", "")
    ) / max(1, n)
    out["mean_words_split"] = sum(t.count("|") for t in split) / max(1, n)
    out["mean_chars_split"] = sum(len(t.replace(" ", "").replace("|", "")) for t in split) / max(1, n)

    # Whitespace granularity of each reference column. FLEURS excludes Khmer from the
    # char-splitting used for CJK/Thai, so a whitespace WER here scores multi-word
    # clauses as single tokens; this measures that granularity directly.
    tok_norm = [t.split() for t in norm]
    tok_raw = [t.replace(ZWSP, " ").split() for t in raw]
    nchars = sum(len(t.replace(" ", "")) for t in norm)
    ntok = sum(len(t) for t in tok_norm)
    out["seg"] = {
        "mean_tokens_norm": ntok / max(1, n),
        "mean_tokens_raw_zwsp": sum(len(t) for t in tok_raw) / max(1, n),
        "mean_chars_per_token_norm": nchars / max(1, ntok),
        "mean_tokens_clausesplit": sum(t.count("|") + 1 for t in split) / max(1, n),
        "frac_tokens_agree_norm_vs_split": _agree(tok_norm, split),
    }

    # Protocol agreement: how much does the choice of reference column + protocol move CER?
    # Measured as the character length ratio, i.e. the denominator of CER.
    out["denominator"] = {
        p.name: sum(len(p(t)) for t in raw) / max(1, sum(len(p(t)) for t in norm))
        for p in ALL_PROTOCOLS
    }
    out["ref_chars"] = {
        "raw_total": sum(len(t) for t in raw),
        "norm_total": sum(len(t) for t in norm),
        "raw_protocols": {p.name: sum(len(p(t)) for t in raw) for p in ALL_PROTOCOLS},
        "norm_protocols": {p.name: sum(len(p(t)) for t in norm) for p in ALL_PROTOCOLS},
    }
    return out


def _name(c: str) -> str:
    try:
        return unicodedata.name(c)
    except ValueError:
        return f"U+{ord(c):04X}"


def sensitivity(jsonls: list[Path]) -> dict:
    """CER of a fixed hypothesis set against both reference columns x all protocols.

    This isolates the reporting artefact: the model output never changes, only the
    reference column and normalisation protocol do.  Whitespace counts are recorded
    alongside, because W1 is the only protocol that scores spaces at all and a
    space-riddled hypothesis is therefore scored differently by W1 alone.
    """
    res = {}
    for jl in jsonls:
        recs = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
        table, spread = {}, {}
        for col in ("raw_ref", "norm_ref"):
            for p in ALL_PROTOCOLS:
                num = den = 0
                for r in recs:
                    ref = p(r[col])
                    hyp = p(r["hyp"])
                    num += edit_distance(ref, hyp)
                    den += len(ref)
                table[f"{col}/{p.name}"] = round(100.0 * num / max(1, den), 3)
            vals = [table[f"{col}/{p.name}"] for p in ALL_PROTOCOLS]
            spread[col] = round(max(vals) - min(vals), 3)
        allv = list(table.values())
        char_cells = [v for k, v in table.items() if not k.endswith("W1_words")]
        n = len(recs)
        spaces = {
            "hyp": sum(W1(r["hyp"]).count(" ") for r in recs) / n,
            "raw_ref": sum(W1(r["raw_ref"]).count(" ") for r in recs) / n,
            "norm_ref": sum(W1(r["norm_ref"]).count(" ") for r in recs) / n,
        }
        res[jl.stem] = {
            "n": n,
            "cer": table,
            "spread_within_column": spread,
            "spread_overall": round(max(allv) - min(allv), 3),
            "spread_char_protocols_only": round(max(char_cells) - min(char_cells), 3),
            "mean_spaces_per_utt": {k: round(v, 2) for k, v in spaces.items()},
        }
        print(f"\n--- {jl.stem} (n={n}) ---", flush=True)
        for k, v in table.items():
            print(f"  {k:28s} CER={v:8.2f}%", flush=True)
        print(f"  spread within raw={spread['raw_ref']}  within norm={spread['norm_ref']}  "
              f"overall={res[jl.stem]['spread_overall']}  "
              f"char-only={res[jl.stem]['spread_char_protocols_only']}", flush=True)
        print(f"  spaces/utt: hyp={spaces['hyp']:.2f} raw_ref={spaces['raw_ref']:.2f} "
              f"norm_ref={spaces['norm_ref']:.2f}", flush=True)
    return res


def backfill_lengths(jsonls: list[Path]) -> int:
    """Add corpus character counts to already-finished bench summaries.

    ``bench.py`` did not originally serialise the CER numerator's companion counts, so
    the four dev runs that establish the Section 4.1 band lack them.  The ``.jsonl``
    sidecar is the primary record of the same run, so recomputing from it is a derivation
    rather than a new measurement; the hypotheses are not regenerated.
    """
    patched = 0
    for jl in jsonls:
        summary = jl.with_suffix(".json")
        if not summary.exists():
            print(f"[skip] {summary.name}: no summary", flush=True)
            continue
        d = json.loads(summary.read_text(encoding="utf-8"))
        if d.get("hyp_chars") is not None:
            print(f"[ok]   {summary.name}: already has lengths", flush=True)
            continue
        recs = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
        ref = sum(len(PRIMARY(r["norm_ref"])) for r in recs)
        hyp = sum(len(PRIMARY(r["hyp"])) for r in recs)
        d["ref_chars"], d["hyp_chars"] = ref, hyp
        d["mean_ref_chars"] = ref / len(recs) if recs else None
        d["mean_hyp_chars"] = hyp / len(recs) if recs else None
        summary.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[fix]  {summary.name}: ref_chars={ref} hyp_chars={hyp} "
              f"mean_hyp={d['mean_hyp_chars']:.1f}", flush=True)
        patched += 1
    return patched


def backfill_ci(jsonls: list[Path], boot: int = 2000) -> int:
    """Attach the percentile bootstrap CI to dev runs that were benched with --boot 0.

    ``aggregate`` resamples sums of already-computed per-utterance edit and reference
    counts, so the interval is a deterministic function of the sidecar and the seed.
    Recomputing it reproduces what ``bench.py --boot 2000`` would have written; the
    hypotheses are not regenerated.
    """
    from khmerasr.metrics import BOOT_SEED, aggregate

    patched = 0
    for jl in jsonls:
        summary = jl.with_suffix(".json")
        if not summary.exists():
            print(f"[skip] {summary.name}: no summary", flush=True)
            continue
        d = json.loads(summary.read_text(encoding="utf-8"))
        touched = False
        recs = None
        if d.get("cer_ci"):
            print(f"[ok]   {summary.name}: already has cer_ci={d['cer_ci']}", flush=True)
            if d.get("boot_resamples") is None:
                # Recover the resample count by reproduction: the interval is a
                # deterministic function of the sidecar, the count and the seed, so the
                # count that regenerates the stored interval bit-for-bit is the one used.
                recs = recs or [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
                pairs = [(r["norm_ref"], r["hyp"]) for r in recs]
                for b in (boot, 1000, 500, 100):
                    ci = [x * 100 for x in aggregate(PRIMARY, pairs, boot=b).cer_ci]
                    if ci == d["cer_ci"]:
                        d["boot_resamples"] = b
                        print(f"[fix]  {summary.name}: boot_resamples={b} reproduced the "
                              "stored interval exactly", flush=True)
                        touched = True
                        break
                else:
                    print(f"[err]  {summary.name}: no resample count reproduced the stored "
                          "interval; leaving boot_resamples unset", flush=True)
        else:
            recs = [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
            cs = aggregate(PRIMARY, [(r["norm_ref"], r["hyp"]) for r in recs], boot=boot)
            if abs(cs.cer * 100 - d["cer"]) > 1e-6:
                print(f"[err]  {summary.name}: recomputed CER {cs.cer*100:.6f} != stored "
                      f"{d['cer']:.6f}; sidecar does not match the run", flush=True)
                continue
            d["cer_ci"] = [x * 100 for x in cs.cer_ci]
            d["boot_resamples"] = boot
            print(f"[fix]  {summary.name}: cer={d['cer']:.3f} ci=[{d['cer_ci'][0]:.3f}, "
                  f"{d['cer_ci'][1]:.3f}] boot={boot} n={cs.n}", flush=True)
            patched += 1
            touched = True
        if d.get("boot_seed") != BOOT_SEED:
            d["boot_seed"] = BOOT_SEED
            touched = True
        if touched:
            summary.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return patched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "results" / "refcols.json"))
    ap.add_argument("--hyp", nargs="*", default=None,
                    help="bench.py .jsonl outputs to run the column x protocol sensitivity on")
    ap.add_argument("--backfill-lengths", action="store_true",
                    help="also add ref_chars/hyp_chars to those runs' summary files")
    ap.add_argument("--backfill-ci", action="store_true",
                    help="also add cer_ci to those runs' summary files")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for --backfill-ci")
    ap.add_argument("--splits", action="store_true",
                    help="recompute the per-split reference-column analysis and rewrite --json")
    args = ap.parse_args()

    if args.hyp and (args.backfill_lengths or args.backfill_ci):
        if args.backfill_lengths:
            backfill_lengths([Path(h) for h in args.hyp])
        if args.backfill_ci:
            backfill_ci([Path(h) for h in args.hyp], boot=args.boot)

    if args.splits:
        res = {}
        for name, rel in SPLITS.items():
            p = ROOT / rel
            if not p.exists():
                print(f"[skip] {name}: {rel} missing", flush=True)
                continue
            res[name] = analyse(p)
            r = res[name]
            print(f"\n=== {name} (n={r['n']}) ===", flush=True)
            print(f"  raw==norm on {r['frac_raw_eq_norm']*100:.2f}% of rows", flush=True)
            print(f"  ZWSP present: raw={r['zwsp_rows_raw']} norm={r['zwsp_rows_norm']} "
                  f"charsplit={r['zwsp_rows_split']}", flush=True)
            print(f"  mean chars: raw={r['mean_len_raw']:.1f} norm={r['mean_len_norm']:.1f}", flush=True)
            if "n_diff" in r:
                print(f"  differing rows={r['n_diff']} "
                      f"only_punct={r['frac_diff_only_punct']*100:.1f}% "
                      f"only_trailing={r['frac_diff_only_trailing']*100:.1f}% "
                      f"mean_editdist={r['mean_editdist_diff']:.2f}", flush=True)
                print(f"  chars only in raw: "
                      + ", ".join(f"{v['name']}x{v['n']}" for v in list(r["chars_only_in_raw"].values())[:5]),
                      flush=True)
            print(f"  charsplit reconstructs norm: {r['frac_split_reconstructs_norm']*100:.2f}%; "
                  f"mean | separators={r['mean_words_split']:.1f}", flush=True)
            s = r["seg"]
            print(f"  seg: tokens(norm)={s['mean_tokens_norm']:.2f} "
                  f"tokens(raw+ZWSP)={s['mean_tokens_raw_zwsp']:.2f} "
                  f"chars/token={s['mean_chars_per_token_norm']:.2f} "
                  f"clauses={s['mean_tokens_clausesplit']:.2f} "
                  f"norm==split_segs={s['frac_tokens_agree_norm_vs_split']*100:.1f}%", flush=True)
            print(f"  CER denominator ratio (raw/norm) per protocol: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in r["denominator"].items()), flush=True)

        if args.hyp:
            res["_sensitivity"] = sensitivity([Path(h) for h in args.hyp])

        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
