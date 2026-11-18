"""Does training-free repetition suppression repair Whisper's Khmer decoder?

Sweeps decoding constraints on a fixed utterance subset and reports CER plus
loop/degeneracy statistics for each configuration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoProcessor, WhisperForConditionalGeneration  # noqa: E402

from khmerasr.corpus import load_fleurs  # noqa: E402
from khmerasr.metrics import repetition_ratio  # noqa: E402
from khmerasr.textnorm import P3  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai-mirror--whisper-large-v3-turbo"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
DTYPE = sys.argv[3] if len(sys.argv) > 3 else "float16"

path = ROOT / "models/models" / MODEL / "snapshots/master"
proc = AutoProcessor.from_pretrained(path)
model = WhisperForConditionalGeneration.from_pretrained(
    path, dtype={"float16": torch.float16, "float32": torch.float32}[DTYPE]
).to("cuda").eval()

utts = load_fleurs(ROOT / "data/fleurs/km_test.tsv", ROOT / "data/fleurs/test/test")[:N]
print(f"# {MODEL} {DTYPE} n={len(utts)}", flush=True)

feats_cache = []
for u in utts:
    audio, sr = sf.read(str(u.wav), dtype="float32")
    f = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", model.dtype)
    feats_cache.append(f)

CONFIGS = {
    "greedy": {},
    "greedy+nrn3": dict(no_repeat_ngram_size=3),
    "greedy+nrn4": dict(no_repeat_ngram_size=4),
    "greedy+rp1.2": dict(repetition_penalty=1.2),
    "greedy+rp1.5": dict(repetition_penalty=1.5),
    "greedy+nrn3+rp1.2": dict(no_repeat_ngram_size=3, repetition_penalty=1.2),
    "greedy+nrn4+rp1.1": dict(no_repeat_ngram_size=4, repetition_penalty=1.1),
    "beam5": dict(num_beams=5),
    "beam5+nrn3": dict(num_beams=5, no_repeat_ngram_size=3),
}

results = {}
for name, extra in CONFIGS.items():
    kw = dict(language="km", task="transcribe", max_new_tokens=440, do_sample=False, num_beams=1)
    if "num_beams" in extra:
        kw["do_sample"] = None
    kw.update(extra)
    kw = {k: v for k, v in kw.items() if v is not None}
    pairs, loops, ntoks, lps = [], [], [], []
    t0 = time.perf_counter()
    for f in feats_cache:
        with torch.inference_mode():
            out = model.generate(f, return_dict_in_generate=True, output_scores=True, **kw)
        text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        ss = getattr(out, "sequences_scores", None)
        lps.append(float(ss[0]) if ss is not None and len(ss) else float("nan"))
        ntoks.append(len(out.sequences[0]) - 1)
        loops.append(repetition_ratio(text))
        pairs.append(text)
    dt = time.perf_counter() - t0

    ed = rc = 0
    for u, h in zip(utts, pairs):
        r, hh = P3(u.norm_ref), P3(h)
        from khmerasr.metrics import edit_distance

        ed += edit_distance(r, hh)
        rc += len(r)
    results[name] = {
        "cer": ed / rc * 100,
        "loop_frac": sum(x > 0.5 for x in loops) / len(loops),
        "mean_rep": sum(loops) / len(loops),
        "mean_ntok": sum(ntoks) / len(ntoks),
        "mean_lp": sum(x for x in lps if x == x) / max(1, sum(x == x for x in lps)),
        "wall_s": dt,
    }
    print(f"{name:22s} CER={results[name]['cer']:6.2f}%  loop={results[name]['loop_frac']*100:5.1f}%  "
          f"rep={results[name]['mean_rep']:.3f}  ntok={results[name]['mean_ntok']:6.1f}  "
          f"lp={results[name]['mean_lp']:6.2f}  {dt:.0f}s", flush=True)

out = ROOT / "results" / f"repsuppress_{MODEL}_{DTYPE}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"model": MODEL, "dtype": DTYPE, "n": len(utts), "configs": results}, indent=2), encoding="utf-8")
print(f"\nwrote {out}", flush=True)

# show the best config's hypotheses for qualitative inspection
best = min(results, key=lambda k: results[k]["cer"])
print(f"\n=== best config: {best} (CER {results[best]['cer']:.2f}%) ===", flush=True)
kw = dict(language="km", task="transcribe", max_new_tokens=440, do_sample=False, num_beams=1)
kw.update(CONFIGS[best])
for u, f in list(zip(utts, feats_cache))[:4]:
    with torch.inference_mode():
        out = model.generate(f, **kw)
    print(f"\nREF: {u.norm_ref[:130]}", flush=True)
    print(f"HYP: {proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()[:130]}", flush=True)
