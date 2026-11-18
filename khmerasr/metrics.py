"""CER/WER and degeneracy diagnostics for low-resource ASR evaluation."""
from __future__ import annotations

import random
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .textnorm import Protocol

KHMER = "KHMER"


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over Unicode codepoints."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def script_of(ch: str) -> str:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNKNOWN"


def script_profile(text: str) -> Counter:
    """Count codepoints by Unicode script name, ignoring punctuation and control/format marks."""
    out: Counter = Counter()
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in "PCZ":
            continue
        out[script_of(ch)] += 1
    return out


def khmer_share(text: str) -> float:
    prof = script_profile(text)
    total = sum(prof.values())
    return prof[KHMER] / total if total else 0.0


def repetition_ratio(text: str, n: int = 10) -> float:
    """1 - distinct/total over character n-grams; high means looping."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 2 * n:
        return 0.0
    grams = [tuple(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


@dataclass
class UtteranceScore:
    uid: str
    ref_chars: int
    hyp_chars: int
    edits: int
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    khmer_share_hyp: float = 1.0
    repetition: float = 0.0
    repetition6: float = 0.0
    empty: bool = False
    duration: float = 0.0
    wall: float = 0.0

    @property
    def cer(self) -> float:
        return self.edits / self.ref_chars if self.ref_chars else 0.0


def score_pair(ref: str, hyp: str, uid: str = "", **kw) -> UtteranceScore:
    s, i, d = edit_ops(ref, hyp)
    return UtteranceScore(
        uid=uid,
        ref_chars=len(ref),
        hyp_chars=len(hyp),
        edits=s + i + d,
        substitutions=s,
        insertions=i,
        deletions=d,
        khmer_share_hyp=khmer_share(hyp),
        repetition=repetition_ratio(hyp, 10),
        repetition6=repetition_ratio(hyp, 6),
        empty=not hyp.strip(),
        **kw,
    )


def edit_ops(a: str, b: str) -> tuple[int, int, int]:
    """Return (substitutions, insertions, deletions) turning ``a`` into ``b``."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prow = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prow[j] + 1, row[j - 1] + 1, prow[j - 1] + (ai != b[j - 1]))
    i, j = n, m
    sub = ins = dele = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, ins, dele


@dataclass
class CorpusScore:
    protocol: str
    n: int
    ref_chars: int
    hyp_chars: int
    edits: int
    substitutions: int
    insertions: int
    deletions: int
    cer: float
    cer_ci: tuple[float, float] | None = None
    frac_empty: float = 0.0
    frac_repetition: float = 0.0
    frac_repetition6: float = 0.0
    mean_repetition6: float = 0.0
    frac_language_drift: float = 0.0
    mean_khmer_share: float = 1.0
    frac_insertion_blowup: float = 0.0
    extras: dict = field(default_factory=dict)


BOOT_SEED = 20261031


def aggregate(protocol: Protocol, pairs: list[tuple[str, str]], boot: int = 0, seed: int = BOOT_SEED) -> CorpusScore:
    """Micro-averaged CER over (raw_ref, raw_hyp) pairs, plus degeneracy rates."""
    refs = [protocol(r) for r, _ in pairs]
    hyps = [protocol(h) for _, h in pairs]

    per: list[UtteranceScore] = []
    for k, (rr, rh) in enumerate(pairs):
        sc = score_pair(refs[k], hyps[k], uid=str(k))
        sc.khmer_share_hyp = khmer_share(rh)
        per.append(sc)

    rc = sum(p.ref_chars for p in per)
    hc = sum(p.hyp_chars for p in per)
    ed = sum(p.edits for p in per)
    n = len(per)
    out = CorpusScore(
        protocol=protocol.name,
        n=n,
        ref_chars=rc,
        hyp_chars=hc,
        edits=ed,
        substitutions=sum(p.substitutions for p in per),
        insertions=sum(p.insertions for p in per),
        deletions=sum(p.deletions for p in per),
        cer=ed / rc if rc else 0.0,
        frac_empty=sum(p.empty for p in per) / n if n else 0.0,
        frac_repetition=sum(p.repetition > 0.5 for p in per) / n if n else 0.0,
        frac_repetition6=sum(p.repetition6 > 0.35 for p in per) / n if n else 0.0,
        mean_repetition6=sum(p.repetition6 for p in per) / n if n else 0.0,
        frac_language_drift=sum(p.khmer_share_hyp < 0.5 for p in per) / n if n else 0.0,
        mean_khmer_share=sum(p.khmer_share_hyp for p in per) / n if n else 0.0,
        frac_insertion_blowup=sum(p.hyp_chars > 2 * p.ref_chars + 20 for p in per) / n if n else 0.0,
    )
    if boot and n:
        rng = random.Random(seed)
        vals = []
        idx = list(range(n))
        for _ in range(boot):
            samp = [per[rng.choice(idx)] for _ in idx]
            s_rc = sum(p.ref_chars for p in samp)
            s_ed = sum(p.edits for p in samp)
            vals.append(s_ed / s_rc if s_rc else 0.0)
        vals.sort()
        out.cer_ci = (vals[int(0.025 * boot)], vals[int(0.975 * boot)])
    return out


def wer_space(refs: list[str], hyps: list[str]) -> float:
    """Space-delimited WER (only meaningful where the script uses spaces)."""
    e = r = 0
    for a, b in zip(refs, hyps):
        ta, tb = a.split(), b.split()
        e += edit_distance_list(ta, tb)
        r += len(ta)
    return e / r if r else 0.0


def edit_distance_list(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
