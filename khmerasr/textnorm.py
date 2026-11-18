"""Khmer-aware text normalization for ASR evaluation.

FLEURS ships two transcription columns per utterance and they disagree on more
than 99% of Khmer rows: ``raw_transcription`` keeps U+200B ZERO WIDTH SPACE as
an informal word separator and retains parenthesised non-speech annotations,
while ``transcription`` replaces ZWSP with ordinary spaces and normalises
punctuation.  Because CER is computed over codepoints, the choice of column and
the treatment of ZWSP and punctuation change the reported score.  Every
normalization used in this project is therefore named and explicit.
"""
from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Callable

ZWSP = "\u200b"

# Whitespace and invisibles that carry no phonetic content.
_INVISIBLE = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00a0\t\r\n"), None
)

# Khmer and ASCII punctuation, plus the Khmer coeng-independent marks that are
# written but never pronounced as separate codepoints.
_PUNCT_CHARS = "។៕៙៚៛" + string.punctuation + "\u2018\u2019\u201c\u201d\u2013\u2014"
_PUNCT = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")
_SPACE = re.compile(r"\s+")

# Parenthesised editorial material in FLEURS raw_transcription, e.g. (ចម្រៀង)
_PAREN = re.compile(r"\([^()]*\)")


def strip_invisible(s: str) -> str:
    return s.translate(_INVISIBLE)


def strip_paren(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _PAREN.sub(" ", s)
    return s


def strip_punct(s: str) -> str:
    return _PUNCT.sub(" ", s)


def collapse_space(s: str) -> str:
    return _SPACE.sub(" ", s).strip()


def remove_space(s: str) -> str:
    return _SPACE.sub("", s)


def casefold(s: str) -> str:
    return s.casefold()


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


@dataclass(frozen=True)
class Protocol:
    """A named, reproducible normalization pipeline."""

    name: str
    steps: tuple[str, ...]
    description: str

    def __call__(self, text: str) -> str:
        return apply_protocol(self, text)


_STEPS: dict[str, Callable[[str], str]] = {
    "nfc": nfc,
    "strip_paren": strip_paren,
    "strip_invisible": strip_invisible,
    "strip_punct": strip_punct,
    "collapse_space": collapse_space,
    "remove_space": remove_space,
    "casefold": casefold,
}


def apply_protocol(proto: Protocol, text: str) -> str:
    for step in proto.steps:
        text = _STEPS[step](text)
    return text


# --- the protocols reported in the paper -----------------------------------

# P0: use the raw column verbatim, only removing invisibles.  This is what an
# evaluator who does not notice the ZWSP convention effectively measures.
P0 = Protocol(
    name="P0_raw",
    steps=("strip_invisible", "remove_space"),
    description="raw_transcription, invisibles dropped, all whitespace removed",
)

# P1: raw column with editorial parentheticals removed.
P1 = Protocol(
    name="P1_raw_noparen",
    steps=("strip_paren", "strip_invisible", "remove_space"),
    description="raw_transcription, parenthesised annotations and invisibles dropped",
)

# P2: normalised column, whitespace removed.  The most common convention.
P2 = Protocol(
    name="P2_norm",
    steps=("strip_invisible", "remove_space"),
    description="transcription column, all whitespace removed",
)

# P3: normalised column, punctuation also removed.  Scores the phonetic
# content only and is the fairest target for a model that does not emit
# Khmer punctuation consistently.
P3 = Protocol(
    name="P3_norm_nopunct",
    steps=("strip_paren", "strip_invisible", "strip_punct", "remove_space"),
    description="transcription column, punctuation and whitespace removed",
)

# P4: same as P3 plus casefolding, needed when a hypothesis drifts into Latin.
P4 = Protocol(
    name="P4_norm_nopunct_cf",
    steps=("strip_paren", "strip_invisible", "strip_punct", "remove_space", "casefold"),
    description="P3 plus Unicode casefolding",
)

# W1: word-level.  The normalised column split on spaces gives a gold Khmer
# word segmentation; a hypothesis must be segmented the same way to be scored.
W1 = Protocol(
    name="W1_words",
    steps=("strip_paren", "strip_invisible", "strip_punct", "collapse_space"),
    description="transcription column, whitespace-collapsed, punctuation removed",
)

ALL_PROTOCOLS = (P0, P1, P2, P3, P4, W1)
PROTOCOL_BY_NAME = {p.name: p for p in ALL_PROTOCOLS}
PRIMARY = P3
