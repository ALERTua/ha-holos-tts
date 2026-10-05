import pytest

from holos_tts.text import (
    MAX_SENTENCE_CHARS,
    STRESS,
    group_sentences,
    needs_verbalization,
    normalize_stress_marks,
    prenormalize,
    prepare_sentences,
    recover_stress,
    split_long_sentence,
    split_sentences,
    strip_stress,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Му+дрого", f"Му{STRESS}дрого"),
        ("русн`я", f"русня{STRESS}"),
        ("Русн+я", f"Русня{STRESS}"),
        ("Ру+сня", f"Ру{STRESS}сня"),
        ("Зустр`іч", f"Зустрі{STRESS}ч"),
        ("2+2=4", "2+2=4"),
        ("a`b", "ab"),
    ],
)
def test_normalize_stress_marks(raw, expected):
    assert normalize_stress_marks(raw) == expected


def test_strip_stress():
    assert strip_stress(f"Му{STRESS}дрого") == "Мудрого"


def test_split_sentences_keeps_end_punctuation():
    assert split_sentences("Перше речення. Друге?  Третє! Четверте: так") == [
        "Перше речення.",
        "Друге?",
        "Третє!",
        "Четверте:",
        "так",
    ]


def test_split_sentences_ends_a_line_without_punctuation():
    assert split_sentences("Привіт\nсвіт.\nЩе рядок") == [
        "Привіт.",
        "світ.",
        "Ще рядок",
    ]


def test_split_sentences_keeps_numbers_with_dots_together():
    assert split_sentences("Версія 1.5 вийшла. Ура") == ["Версія 1.5 вийшла.", "Ура"]


def test_group_sentences_closes_a_chunk_after_the_limit():
    sentences = ["a" * 30, "b" * 30, "c" * 50, "d" * 10]
    assert group_sentences(sentences, 100) == [
        ["a" * 30, "b" * 30, "c" * 50],
        ["d" * 10],
    ]


def test_group_sentences_keeps_a_long_sentence_whole():
    assert group_sentences(["x" * 300, "y"], 100) == [["x" * 300], ["y"]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("о 15:30", True),
        ("вологість 87%", True),
        ("кімната №3", True),
        ("датчик CO", True),
        ("ДСНС повідомляє", True),
        ("Невеличкі котеджі обабіч.", False),
        (f"Русня{STRESS} вже майже вся.", False),
        ("Іван пішов.", False),
    ],
)
def test_needs_verbalization(text, expected):
    assert needs_verbalization(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Зустріч 12.05.2025 о 10.", "Зустріч 12 травня 2025 року о 10."),
        ("Дата 31.13.2025 погана.", "Дата 31.13.2025 погана."),
        ("Версія 1.5 вийшла.", "Версія 1.5 вийшла."),
    ],
)
def test_prenormalize_writes_the_month_of_a_numeric_date(text, expected):
    assert prenormalize(text) == expected


def test_recover_stress_keeps_marks_on_unchanged_words():
    original = f"Зустрі{STRESS}ч о 7 ранку."
    verbalized = "Зустріч о сьомій ранку."
    assert recover_stress(original, verbalized) == f"Зустрі{STRESS}ч о сьомій ранку."


def test_recover_stress_without_marks_returns_the_verbalized_text():
    assert recover_stress("о 7 ранку", "о сьомій ранку") == "о сьомій ранку"


def test_prepare_sentences_runs_every_step():
    assert prepare_sentences("Русн+я? Зустріч 01.02.2026") == [
        f"Русня{STRESS}?",
        "Зустріч 1 лютого 2026 року",
    ]


def test_split_long_sentence_at_commas_then_spaces():
    sentence = ", ".join(["слово " * 20] * 3).strip()
    pieces = split_long_sentence(sentence, limit=150)
    assert len(pieces) > 1
    assert all(len(piece) <= 150 for piece in pieces)
    assert " ".join(pieces).split() == sentence.split()
    assert pieces[0].endswith(",")


def test_split_long_sentence_keeps_a_short_sentence():
    assert split_long_sentence("Коротке речення.", limit=150) == ["Коротке речення."]


def test_prepare_sentences_splits_a_long_sentence():
    assert all(len(piece) <= MAX_SENTENCE_CHARS for piece in prepare_sentences("дуже довге " * 60))


def test_windows_line_breaks_end_a_sentence():
    assert split_sentences("Привіт\r\nСвіт") == ["Привіт.", "Світ"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("сім'+я", f"сім'я{STRESS}"), ("сім’+я", f"сім’я{STRESS}"), ("2+а", "2+а"), (" +а", " +а")],
)
def test_plus_before_a_vowel_needs_a_letter_or_an_apostrophe(raw, expected):
    assert normalize_stress_marks(raw) == expected


@pytest.mark.parametrize("text", ["10 м²", "½ склянки", "Карл Ⅳ"])
def test_needs_verbalization_for_number_symbols(text):
    assert needs_verbalization(text)


def test_split_long_sentence_keeps_a_number_with_spaces_whole():
    sentence = "слово " * 23 + "1 000 000 гривень"
    pieces = split_long_sentence(sentence, limit=150)
    assert any("1 000 000" in piece for piece in pieces)
