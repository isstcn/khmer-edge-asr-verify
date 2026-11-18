"""FLEURS / OpenSLR manifest loading for the Khmer ASR benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Utterance:
    uid: str
    wav: Path
    raw_ref: str  # raw_transcription: ZWSP word separators, parentheticals kept
    norm_ref: str  # transcription: normalised whitespace and punctuation
    n_samples: int
    gender: str

    @property
    def duration(self) -> float:
        return self.n_samples / 16000.0


def load_fleurs(tsv: Path, wavdir: Path) -> list[Utterance]:
    out = []
    for line in Path(tsv).read_text(encoding="utf-8").splitlines():
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        out.append(
            Utterance(
                uid=f"{p[0]}_{Path(p[1]).stem}",
                wav=Path(wavdir) / p[1],
                raw_ref=p[2],
                norm_ref=p[3],
                n_samples=int(p[5]),
                gender=p[6],
            )
        )
    return out


def load_slr42(root: Path) -> list[Utterance]:
    """OpenSLR SLR42 (Khmer male TTS-prompt recordings, CC BY-SA 4.0).

    Verified layout after unzipping ``km_kh_male.zip``::

        <root>/km_kh_male/line_index.tsv      uid <TAB> <TAB> transcript
        <root>/km_kh_male/wavs/<uid>.wav      2,906 files

    Unlike FLEURS, whose Khmer whitespace marks clause boundaries, the SLR42
    transcripts are segmented into single words, so they support a genuine
    word-level error rate. There is no second reference column and no
    punctuation, hence ``raw_ref == norm_ref``.
    """
    root = Path(root)
    index = root / "km_kh_male" / "line_index.tsv"
    wavdir = root / "km_kh_male" / "wavs"
    if not index.exists():
        cands = sorted(root.rglob("line_index.tsv"))
        if not cands:
            return []
        index = cands[0]
        wavdir = index.parent / "wavs"

    import wave

    out = []
    for line in index.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        uid = parts[0].strip()
        text = parts[-1].strip()
        if not uid or not text:
            continue
        wav = wavdir / f"{uid}.wav"
        if not wav.exists():
            continue
        with wave.open(str(wav), "rb") as w:
            n = w.getnframes()
        out.append(
            Utterance(
                uid=uid,
                wav=wav,
                raw_ref=text,
                norm_ref=text,
                n_samples=n,
                gender="MALE",
            )
        )
    return out


def hours(us: list[Utterance]) -> float:
    return sum(u.duration for u in us) / 3600.0
