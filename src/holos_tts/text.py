"""
Text steps that need no model: stress marks, sentence splitting and the verbalizer filter.

The server keeps a stress mark as the combining acute accent (U+0301) right after the stressed vowel.
That is the form that the stress model and the phonemizer expect.
"""

from __future__ import annotations

import difflib
import re

STRESS = "́"
VOWELS = "аеєиіїоуюяАЕЄИІЇОУЮЯ"

# A stress is "+" after a vowel, "+" between a letter and a vowel (as in "Русн+я"), or "`" before a vowel
_PLUS_AFTER_VOWEL = re.compile(rf"([{VOWELS}])\+")
_PLUS_BEFORE_VOWEL = re.compile(rf"(?<=[^\W\d_]|['’ʼ])\+([{VOWELS}])")
_BACKTICK_BEFORE_VOWEL = re.compile(rf"`([{VOWELS}])")

# A sentence ends at one of these marks when a space follows. A line break also ends a sentence.
SENTENCE_END_MARKS = ".?!:…"
_SENTENCE_END = re.compile(rf"(?<=[{SENTENCE_END_MARKS}])\s+")
_LINE_WITHOUT_END = re.compile(r"(\w[^.,!:?…\-\s]?)[ \t]*\n+")

# The verbalizer rewrites numbers, symbols, Latin words and acronyms. Other text goes to the stress model as it is,
# because the verbalizer is slow and sometimes changes plain words.
_NEEDS_VERBALIZER = re.compile(r"[\d%№°$€£₴§&@#=/<>²³¹¼-¾⁰-₟⅐-↋]|[A-Za-z]|\b[А-ЯІЇЄҐ]{2,}\b")

# a longer sentence splits at commas, then at spaces (engine.split_phonemes keeps each model pass under 25 s)
MAX_SENTENCE_CHARS = 150
_CLAUSE_END = re.compile(r"(?<=[,;—–])\s+")
# a space between two digits belongs to a number such as "1 000 000"
_SPACES = re.compile(r"\s+(?!\d)|(?<!\d)\s+")

_MONTHS = (
    "січня",
    "лютого",
    "березня",
    "квітня",
    "травня",
    "червня",
    "липня",
    "серпня",
    "вересня",
    "жовтня",
    "листопада",
    "грудня",
)
# The verbalizer leaves a numeric date such as 12.05.2025 as it is, so the server writes the month as a word first
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")


def normalize_stress_marks(text: str) -> str:
    """Turn the user stress marks "+" and "`" into the combining acute accent after the vowel."""
    text = _BACKTICK_BEFORE_VOWEL.sub(rf"\1{STRESS}", text)
    text = _PLUS_AFTER_VOWEL.sub(rf"\1{STRESS}", text)
    text = _PLUS_BEFORE_VOWEL.sub(rf"\1{STRESS}", text)
    return text.replace("`", "")


def strip_stress(text: str) -> str:
    """Remove each stress mark."""
    return text.replace(STRESS, "")


def _date_to_words(match: re.Match[str]) -> str:
    day, month, year = int(match[1]), int(match[2]), match[3]
    if not (1 <= day <= 31 and 1 <= month <= 12):  # noqa: PLR2004 - calendar limits
        return match[0]

    return f"{day} {_MONTHS[month - 1]} {year} року"


def prenormalize(text: str) -> str:
    """Rewrite the text forms that the verbalizer does not handle."""
    return _NUMERIC_DATE.sub(_date_to_words, text)


def split_sentences(text: str) -> list[str]:
    """Split the text into sentences. A line without end punctuation becomes a sentence of its own."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _LINE_WITHOUT_END.sub(r"\1. ", text.strip())
    text = " ".join(text.split())
    return [sentence for sentence in _SENTENCE_END.split(text) if sentence.strip()]


def _join_up_to(parts: list[str], limit: int) -> list[str]:
    pieces: list[str] = []
    for part in parts:
        if pieces and len(pieces[-1]) + 1 + len(part) <= limit:
            pieces[-1] += " " + part
        else:
            pieces.append(part)

    return pieces


def split_long_sentence(sentence: str, limit: int = MAX_SENTENCE_CHARS) -> list[str]:
    """Split a sentence longer than ``limit`` characters at commas, then at spaces."""
    if len(sentence) <= limit:
        return [sentence]

    pieces: list[str] = []
    for clause in _join_up_to(_CLAUSE_END.split(sentence), limit):
        pieces.extend(_join_up_to(_SPACES.split(clause), limit) if len(clause) > limit else [clause])

    return pieces


def group_sentences(sentences: list[str], limit: int) -> list[list[str]]:
    """
    Join neighbor sentences into chunks for the acoustic model.

    A chunk takes the next sentence while the chunk is not longer than ``limit`` characters.
    The HolosTTS and StyleTTS2 demos split the text with the same rule.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        current.append(sentence)
        length += len(sentence) + (1 if length else 0)
        if length > limit:
            chunks.append(current)
            current, length = [], 0

    if current:
        chunks.append(current)

    return chunks


def needs_verbalization(text: str) -> bool:
    """Tell if the text has a number, a symbol, a Latin word or an acronym that the verbalizer must rewrite."""
    return bool(_NEEDS_VERBALIZER.search(strip_stress(text)))


def recover_stress(original: str, verbalized: str) -> str:
    """
    Put the stress marks of ``original`` back into ``verbalized``.

    The verbalizer gets the text without stress marks. Each word that the verbalizer did not change gets its mark
    back. A word that the verbalizer rewrote, for example a number, stays without a mark.
    """
    if STRESS not in original:
        return verbalized

    original_words = original.split()
    plain_words = [strip_stress(word) for word in original_words]
    result = verbalized.split()
    matcher = difflib.SequenceMatcher(None, plain_words, result, autojunk=False)
    for tag, i1, i2, j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                result[j1 + offset] = original_words[i1 + offset]

    return " ".join(result)


def prepare_sentences(text: str) -> list[str]:
    """Run each step of this module that the main process does before it sends the text to the worker."""
    sentences = split_sentences(prenormalize(normalize_stress_marks(text)))
    return [piece for sentence in sentences for piece in split_long_sentence(sentence)]
