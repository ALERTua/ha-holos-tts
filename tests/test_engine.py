import numpy as np
import pytest

from holos_tts.engine import (
    PAD_ID,
    TOKEN_IDS,
    VOCAB,
    VOICE_SIZE,
    load_voice_file,
    order_voices,
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
