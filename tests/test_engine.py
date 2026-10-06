import logging

import numpy as np
import pytest

from holos_tts import engine
from holos_tts.engine import (
    PAD_ID,
    PASS_TOKENS,
    TOKEN_IDS,
    VOCAB,
    VOICE_SIZE,
    HolosEngine,
    load_voice_file,
    order_voices,
    pass_tokens,
    split_phonemes,
    tokenize,
)


def test_vocabulary_matches_the_checkpoint_size():
    assert len(VOCAB) == 184


def test_repeated_symbol_keeps_its_last_index():
    assert TOKEN_IDS["'"] == VOCAB.rindex("'")


def test_tokenize_adds_pads_and_drops_unknown_symbols():
    ids = tokenize("abЖ")
    assert ids == [PAD_ID, TOKEN_IDS["a"], TOKEN_IDS["b"], PAD_ID]


def test_order_voices_puts_the_default_first_and_numbers_in_order():
    names = {"Speaker_10", "Speaker_2", "Гаська Шиян", "Мій голос"}
    assert order_voices(names, "Speaker_2") == [
        "Speaker_2",
        "Гаська Шиян",
        "Мій голос",
        "Speaker_10",
    ]


def test_order_voices_without_the_default():
    assert order_voices({"Speaker_1", "Speaker_0"}, "missing") == [
        "Speaker_0",
        "Speaker_1",
    ]


def test_load_voice_file_reads_npy(tmp_path):
    path = tmp_path / "voice.npy"
    np.save(path, np.ones((1, VOICE_SIZE), dtype=np.float32))
    assert load_voice_file(path).shape == (VOICE_SIZE,)


def test_load_voice_file_refuses_a_wrong_size(tmp_path):
    path = tmp_path / "voice.npy"
    np.save(path, np.ones(10, dtype=np.float32))
    with pytest.raises(ValueError, match="256"):
        load_voice_file(path)


class FakeSession:
    """The graph: one sample for each token at speed 1.0, the sample value is the token id, cut at the audio cap."""

    def __init__(self):
        self.passes = []

    def run(self, _outputs, feeds):
        tokens = feeds["tokens"][0]
        self.passes.append(len(tokens))
        repeat = round(1 / float(feeds["speed"][0]))
        audio = np.repeat(tokens[1:-1].astype(np.float32), repeat)
        return audio[np.newaxis], np.array([min(len(audio), engine.MAX_PASS_SAMPLES)])


@pytest.fixture
def fake_engine():
    holos = HolosEngine.__new__(HolosEngine)
    holos._session = FakeSession()
    holos._voices = {"voice": np.zeros(VOICE_SIZE, dtype=np.float32)}
    return holos


def spoken(text):
    return [float(TOKEN_IDS[symbol]) for symbol in text]


def test_pass_budget_follows_the_speed():
    assert (pass_tokens(0.5), pass_tokens(1.0), pass_tokens(2.0)) == (PASS_TOKENS // 2, PASS_TOKENS, PASS_TOKENS * 2)


def test_split_prefers_a_sentence_end_to_a_more_even_clause_cut():
    assert split_phonemes("abcd efgh, ijkl mnop. qrst uvwx", 30) == ["abcd efgh, ijkl mnop.", "qrst uvwx"]


def test_split_prefers_a_clause_mark_to_a_more_even_word_cut():
    assert split_phonemes("abcd, efgh ijkl mnop qrst uvwx", 30) == ["abcd,", "efgh ijkl mnop qrst uvwx"]


def test_split_makes_even_parts_of_whole_words_in_the_fewest_passes():
    phonemes = " ".join(["ab"] * 100)
    parts = split_phonemes(phonemes, 50)
    sizes = [len(tokenize(part)) for part in parts]
    assert " ".join(parts) == phonemes
    assert all(part.split() == ["ab"] * len(part.split()) for part in parts)
    assert len(parts) == 7
    assert max(sizes) <= 50
    assert max(sizes) - min(sizes) <= 3


def test_a_word_longer_than_the_budget_stays_whole():
    long_word = "ab" * 25
    assert split_phonemes(f"ab {long_word} de", 10) == ["ab", long_word, "de"]


def test_a_chunk_within_the_budget_takes_one_pass(fake_engine):
    phonemes = " ".join(["abcd"] * PASS_TOKENS)[: PASS_TOKENS - 2]
    assert fake_engine.synthesize(phonemes, "voice", 1.0).tolist() == spoken(phonemes)
    assert fake_engine._session.passes == [PASS_TOKENS]


def test_a_slower_speed_splits_the_same_chunk(fake_engine):
    phonemes = " ".join(["abcd"] * 40)
    fake_engine.synthesize(phonemes, "voice", 1.0)
    fake_engine.synthesize(phonemes, "voice", 0.5)
    assert len(fake_engine._session.passes) == 1 + 2


def test_parts_of_a_long_chunk_are_joined_in_order(fake_engine, monkeypatch):
    monkeypatch.setattr(engine, "PASS_TOKENS", 12)
    audio = fake_engine.synthesize("abcd efgh. ijkl mnop. qrst", "voice", 1.0)
    assert audio.tolist() == spoken("abcd efgh.ijkl mnop.qrst")
    assert len(fake_engine._session.passes) == 3


def test_a_pass_at_the_audio_cap_is_spoken_again_in_parts(fake_engine, monkeypatch, caplog):
    monkeypatch.setattr(engine, "MAX_PASS_SAMPLES", 20)
    with caplog.at_level(logging.WARNING, logger=engine.LOG.name):
        audio = fake_engine.synthesize("abcd efgh. ijkl mnop qrst", "voice", 1.0)

    assert audio.tolist() == spoken("abcd efgh.ijkl mnop qrst")
    assert "reached the audio cap of 20 samples, so the server speaks it again" in caplog.text


def test_a_pass_at_the_audio_cap_without_a_space_keeps_its_audio(fake_engine, monkeypatch, caplog):
    monkeypatch.setattr(engine, "MAX_PASS_SAMPLES", 20)
    with caplog.at_level(logging.WARNING, logger=engine.LOG.name):
        audio = fake_engine.synthesize("a" * 30, "voice", 1.0)

    assert len(audio) == 20
    assert "it has no space, so its end is lost" in caplog.text
