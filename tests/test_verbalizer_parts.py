import json
import logging
import re
import types

import pytest

from holos_tts import verbalizer
from holos_tts.frontend import verbalize_sentence
from holos_tts.text import STRESS, governing_preposition, halve_for_verbalizer
from holos_tts.verbalizer import verbalize_in_parts

DIGITS = dict(
    zip(
        "0123456789", ("нуль", "один", "два", "три", "чотири", "пʼять", "шість", "сім", "вісім", "девʼять"), strict=True
    )
)


class FakeModel:
    """Spell each digit, start with a capital letter, end a clause as a sentence, and cut a text over ``limit``."""

    def __init__(self, limit):
        self.limit = limit
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        words = re.sub(r" ([,.;])", r"\1", re.sub(r"\d", lambda match: DIGITS[match[0]] + " ", text)).strip()
        words = words[:1].upper() + words[1:]
        if words[-1] in ",;":
            words = words[:-1] + "."
        if len(text) > self.limit:
            return words[: len(words) // 2].strip(), False

        return words, True


def test_a_text_that_the_model_ends_itself_goes_to_it_once_and_comes_back_as_it_is():
    model = FakeModel(limit=100)

    assert verbalize_in_parts("зараз 7, надворі 3,", model) == "Зараз сім, надворі три."
    assert model.calls == ["зараз 7, надворі 3,"]


def test_a_cut_text_goes_to_the_model_again_in_halves_that_join_into_one_sentence():
    model = FakeModel(limit=20)

    words = verbalize_in_parts("Рахунок 1, 2, 3, 4, 5, 6.", model)

    assert words == "Рахунок один, два, три, чотири, пʼять, шість."
    assert model.calls == ["Рахунок 1, 2, 3, 4, 5, 6.", "Рахунок 1, 2,", "і 3, 4, 5, 6."]


def test_a_later_half_that_starts_with_a_signed_number_also_gets_a_word_before_it():
    model = FakeModel(limit=20)

    assert verbalize_in_parts("Погода +1, +2, +3, +4.", model) == "Погода +один, +два, +три, +чотири."
    assert model.calls[-1] == "і +2, +3, +4."


def test_a_later_half_of_a_number_list_gets_the_preposition_of_the_list_instead_of_i():
    model = FakeModel(limit=24)

    words = verbalize_in_parts("Рейси о 1, 2, 3, 4, 5, 6.", model)

    assert words == "Рейси о один, два, три, чотири, пʼять, шість."
    assert model.calls[1:] == ["Рейси о 1, 2,", "о 3, 4, 5, 6."]


def test_a_preposition_in_an_earlier_part_still_governs_a_later_quarter():
    model = FakeModel(limit=12)

    words = verbalize_in_parts("Рейси о 1, 2, 3, 4, 5, 6, 7, 8.", model)

    assert words == "Рейси о один, два, три, чотири, пʼять, шість, сім, вісім."
    assert model.calls[-2:] == ["о 4, 5,", "о 6, 7, 8."]
    assert not [call for call in model.calls if call.startswith("і ")]


@pytest.mark.parametrize(
    ("text", "preposition"),
    [
        ("Автобуси о 06:15, 06:45,", "о"),
        ("Температура падала до -3°C, -5°C,", "до"),
        ("Знижки 10%, 15% і до 20%,", "до"),
        ("Пороги від 5 до 10, 15,", "до"),
        ("Зустріч О 7:30, 8:00,", "о"),
        ("Відпустка з 1 по 5 березня,", None),
        ("Зустріч о 7:30 у кімнаті 214,", None),
        ("Рахунок 1, 2,", None),
        ("Ріо 5, 6,", None),
        ("", None),
    ],
)
def test_governing_preposition_is_the_last_preposition_before_a_list_of_numbers_only(text, preposition):
    assert governing_preposition(text) == preposition


def test_the_first_half_keeps_the_start_and_the_last_half_keeps_the_end_of_the_text():
    model = FakeModel(limit=12)

    assert verbalize_in_parts("1, 2, 3, 4, 5, 6;", model) == "Один, два, три, чотири, пʼять, шість."
    assert model.calls[1] == "1, 2, 3,"


def test_a_later_half_that_starts_with_a_word_keeps_its_small_letter():
    model = FakeModel(limit=20)

    assert verbalize_in_parts("Ціни 1, 2, вода 3, 4.", model) == "Ціни один, два, вода три, чотири."
    assert model.calls[-1] == "вода 3, 4."


def test_a_half_that_the_model_still_cuts_is_halved_again():
    model = FakeModel(limit=12)

    words = verbalize_in_parts("Рахунок 1, 2, 3, 4, 5, 6, 7, 8.", model)

    assert words == "Рахунок один, два, три, чотири, пʼять, шість, сім, вісім."
    assert len(model.calls) > 3


def test_a_cut_text_without_a_place_to_split_keeps_the_cut_words_and_warns(caplog):
    model = FakeModel(limit=3)

    with caplog.at_level(logging.WARNING, logger="holos_tts.verbalizer"):
        words = verbalize_in_parts("1234", model)

    assert words == "Один два"
    assert model.calls == ["1234"]
    assert "no place to split" in caplog.text


def test_the_stress_marks_of_the_user_come_back_in_each_half():
    model = FakeModel(limit=25)
    sentence = f"Ре{STRESS}йси 1, 2, 3, ві{STRESS}дкладено на 4."

    words = verbalize_sentence(sentence, lambda text: verbalize_in_parts(text, model))

    assert words == f"Ре{STRESS}йси один, два, три, ві{STRESS}дкладено на чотири."
    assert len(model.calls) == 3


@pytest.mark.parametrize(
    ("text", "halves"),
    [
        ("а 1, б 2, в 3, г 4.", ["а 1, б 2,", "в 3, г 4."]),
        ("Так; ні", ["Так;", "ні"]),
        ("Завтра — дощ", ["Завтра —", "дощ"]),
        ("один два три чотири", ["один два", "три чотири"]),
        ("Так, ", ["Так, "]),
        ("Початок, а далі дуже довгий кінець речення", ["Початок,", "а далі дуже довгий кінець речення"]),
    ],
)
def test_halve_splits_at_the_clause_end_nearest_the_middle_else_between_words(text, halves):
    assert halve_for_verbalizer(text) == halves


@pytest.mark.parametrize(
    ("text", "halves"),
    [
        ("Зустріч 22 серпня 2025 року о 15:30", ["Зустріч 22 серпня 2025 року", "о 15:30"]),
        ("Разом 1 000 000 гривень", ["Разом 1 000 000 гривень"]),
        ("Від 10 – 15 °C", ["Від 10 – 15 °C"]),
        ("22.08.2025", ["22.08.2025"]),
        ("о 15:30 -3°C", ["о 15:30 -3°C"]),
    ],
)
def test_halve_never_splits_next_to_a_digit(text, halves):
    assert halve_for_verbalizer(text) == halves


class FakeTranslator:
    """Echo the pieces, or return a hypothesis of ``cut_length`` tokens for a source of more than four pieces."""

    cut_length = 0
    sources: list[list[str]] = []

    def __init__(self, *_, **__):
        pass

    def translate_batch(self, source, **_):
        FakeTranslator.sources.append(source[0])
        pieces = source[0][1:-1]
        hypothesis = [verbalizer.LANGUAGE_TOKEN, *pieces]
        if len(pieces) > 4:
            hypothesis = [verbalizer.LANGUAGE_TOKEN, *["слово"] * (FakeTranslator.cut_length - 1)]

        return [types.SimpleNamespace(hypotheses=[hypothesis])]


class FakeSentencePiece:
    def __init__(self, **_):
        pass

    def encode(self, text, out_type):
        assert out_type is str
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture
def model_verbalizer(tmp_path, monkeypatch):
    FakeTranslator.sources = []
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps(dict.fromkeys(("а,", "б,", "в,", "г,", "ґ,", "д", "слово"), 0)), encoding="utf-8")
    monkeypatch.setattr(verbalizer, "snapshot_download", lambda *_, **__: str(tmp_path))
    monkeypatch.setattr(verbalizer, "hf_hub_download", lambda *_, **__: str(vocab))
    monkeypatch.setattr(verbalizer.sentencepiece, "SentencePieceProcessor", FakeSentencePiece)
    monkeypatch.setattr(verbalizer.ctranslate2, "Translator", FakeTranslator)
    return verbalizer.Verbalizer("cpu", 1, None)


def test_an_output_shorter_than_the_learned_limit_is_complete(model_verbalizer):
    FakeTranslator.cut_length = verbalizer.MAX_OUTPUT_TOKENS - 1

    assert model_verbalizer("а, б, в, г, ґ, д") == " ".join(["слово"] * (verbalizer.MAX_OUTPUT_TOKENS - 2))
    assert len(FakeTranslator.sources) == 1


def test_an_output_of_the_learned_limit_is_cut_and_the_halves_go_to_the_model(model_verbalizer):
    FakeTranslator.cut_length = verbalizer.MAX_OUTPUT_TOKENS

    assert model_verbalizer("а, б, в, г, ґ, д") == "а, б, в, г, ґ, д"
    assert [source[1:-1] for source in FakeTranslator.sources[1:]] == [["а,", "б,", "в,"], ["г,", "ґ,", "д"]]
