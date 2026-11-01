import logging
from pathlib import Path

import numpy as np
import pytest

from holos_tts import engine
from holos_tts.engine import (
    PAD_ID,
    TOKEN_IDS,
    VOCAB,
    VOICE_SIZE,
    HolosEngine,
    load_voice_file,
    load_voices,
    order_voices,
    read_voices_npz,
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


def _write_npz(path: Path, names: list[str], values: object) -> Path:
    np.savez(path, names=np.array(names), values=np.asarray(values, dtype=np.float32))
    return path


def test_read_voices_npz_pairs_the_names_with_the_vectors(tmp_path):
    path = _write_npz(
        tmp_path / "voices.npz", ["Speaker_1", "Гаська Шиян"], [np.zeros(VOICE_SIZE), np.ones(VOICE_SIZE)]
    )

    voices = read_voices_npz(path)

    assert list(voices) == ["Speaker_1", "Гаська Шиян"]
    assert voices["Гаська Шиян"].tolist() == [1.0] * VOICE_SIZE
    assert voices["Speaker_1"].dtype == np.float32


def test_read_voices_npz_refuses_names_without_vectors(tmp_path):
    path = _write_npz(tmp_path / "voices.npz", ["a", "b"], [np.zeros(VOICE_SIZE)])

    with pytest.raises(ValueError, match="2 voice names for 1 voices"):
        read_voices_npz(path)


def test_read_voices_npz_refuses_a_wrong_vector_size(tmp_path):
    path = _write_npz(tmp_path / "voices.npz", ["a"], [np.zeros(10)])

    with pytest.raises(ValueError, match="256"):
        read_voices_npz(path)


def test_load_voices_adds_the_extra_voices_and_skips_broken_files(tmp_path, monkeypatch, caplog):
    model_voices = _write_npz(tmp_path / "voices.npz", ["Speaker_1", "Мій голос"], np.zeros((2, VOICE_SIZE)))
    downloads = []
    monkeypatch.setattr(
        engine, "hf_hub_download", lambda *args, **kwargs: downloads.append((args, kwargs)) or str(model_voices)
    )
    extra = tmp_path / "voices"
    extra.mkdir()
    np.save(extra / "Мій голос.npy", np.ones(VOICE_SIZE, dtype=np.float32))
    np.save(extra / "broken.npy", np.ones(3, dtype=np.float32))
    (extra / "notes.txt").write_text("not a voice", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger=engine.LOG.name):
        voices = load_voices(extra)

    assert downloads == [((engine.MODEL_REPO, engine.VOICES_FILE), {"revision": engine.MODEL_REVISION})]
    assert sorted(voices) == ["Speaker_1", "Мій голос"]
    assert voices["Мій голос"].tolist() == [1.0] * VOICE_SIZE
    assert "Skipping the voice file" in caplog.text


def test_load_voices_without_the_extra_folder(tmp_path, monkeypatch):
    model_voices = _write_npz(tmp_path / "voices.npz", ["Speaker_1"], np.zeros((1, VOICE_SIZE)))
    monkeypatch.setattr(engine, "hf_hub_download", lambda *_args, **_kwargs: str(model_voices))

    assert list(load_voices(tmp_path / "missing")) == ["Speaker_1"]


def test_split_keeps_phonemes_within_the_budget_as_they_are():
    assert split_phonemes(" abcd efgh ", 13) == [" abcd efgh "]


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


# the fake model: each token lasts FRAMES[symbol] frames at speed 1.0 (pads and other symbols one frame)
FRAMES = {TOKEN_IDS["a"]: 3}


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def run(self, outputs, feeds):
        assert outputs == ["acoustic_tokens", "prosodic_tokens", "durations"]
        tokens = feeds["tokens"][0]
        self.calls.append(len(tokens))
        repeat = round(1 / float(feeds["speed"][0]))
        durations = np.array([[FRAMES.get(int(token), 1) * repeat for token in tokens]], dtype=np.int64)
        features = tokens.astype(np.float32)[np.newaxis]
        return features, features, durations


class FakeProsody:
    def run(self, outputs, feeds):
        assert outputs == ["f0", "energy"]
        frames = feeds["frame_to_token"].size
        return np.zeros((1, frames), np.float32), np.zeros((1, frames), np.float32)


class FakeDecoder:
    """One sample for each frame: the id of the token of that frame, without the pads."""

    def __init__(self):
        self.frames = []

    def run(self, outputs, feeds):
        assert outputs == ["audio", "audio_lengths"]
        frame_to_token = feeds["frame_to_token"]
        self.frames.append(frame_to_token.size)
        audio = feeds["acoustic_tokens"][0][frame_to_token]
        audio = audio[audio != PAD_ID]
        padded = np.concatenate([audio, np.full(5, -1, np.float32)])
        return padded[np.newaxis], np.array([audio.size])


@pytest.fixture
def fake_engine():
    holos = HolosEngine.__new__(HolosEngine)
    holos._encoder = FakeEncoder()
    holos._prosody = FakeProsody()
    holos._decoder = FakeDecoder()
    holos._voices = {"voice": np.zeros(VOICE_SIZE, dtype=np.float32)}
    holos._max_frames = 1000
    return holos


def spoken(text, repeat=1):
    return [float(TOKEN_IDS[s]) for s in text if s in TOKEN_IDS for _ in range(FRAMES.get(TOKEN_IDS[s], 1) * repeat)]


def test_a_chunk_within_the_frame_limit_takes_one_pass(fake_engine):
    fake_engine._max_frames = 10

    audio = fake_engine.synthesize("bcd efgh", "voice", 1.0)

    assert audio.tolist() == spoken("bcd efgh")
    assert audio.dtype == np.float32
    assert fake_engine._decoder.frames == [8 + 2]


def test_the_decoder_gets_the_frames_of_the_encoder_durations(fake_engine):
    audio = fake_engine.synthesize("ab", "voice", 1.0)

    assert audio.tolist() == spoken("ab")
    assert fake_engine._decoder.frames == [1 + 3 + 1 + 1]


def test_a_chunk_above_the_frame_limit_is_spoken_in_parts_in_order(fake_engine):
    fake_engine._max_frames = 12

    audio = fake_engine.synthesize("bcde fghi. jklm nopq. rstu", "voice", 1.0)

    assert audio.tolist() == spoken("bcde fghi.jklm nopq.rstu")
    assert len(fake_engine._decoder.frames) > 1
    assert max(fake_engine._decoder.frames) <= 12


def test_a_slower_speed_splits_the_same_chunk(fake_engine):
    fake_engine._max_frames = 24

    fake_engine.synthesize("bcde fghi jklm", "voice", 1.0)
    fake_engine.synthesize("bcde fghi jklm", "voice", 0.5)

    assert fake_engine._decoder.frames == [16, 12, 22]


def test_a_part_with_longer_tokens_is_split_again(fake_engine, caplog):
    fake_engine._max_frames = 17

    with caplog.at_level(logging.WARNING, logger=engine.LOG.name):
        audio = fake_engine.synthesize("aa aaaa fghi", "voice", 1.0)

    # "aa aaaa" fits the token budget, but its "a" tokens last 21 frames, so it is encoded again in two parts
    assert fake_engine._encoder.calls == [14, 9, 4, 6, 6]
    assert fake_engine._decoder.frames == [8, 14, 6]
    assert audio.tolist() == spoken("aaaaaafghi")
    assert not caplog.records


def test_a_word_above_the_frame_limit_is_cut_with_a_warning(fake_engine, caplog):
    fake_engine._max_frames = 10

    with caplog.at_level(logging.WARNING, logger=engine.LOG.name):
        audio = fake_engine.synthesize("b" * 30, "voice", 1.0)

    assert audio.tolist() == spoken("b" * 30)
    assert max(fake_engine._decoder.frames) <= 10
    assert "is longer than 10 frames, so the server cuts it" in caplog.text


def test_the_decoder_never_gets_more_frames_than_its_limit(fake_engine):
    fake_engine._max_frames = 4
    encoded = fake_engine._encode("bcde", fake_engine._voices["voice"].reshape(1, VOICE_SIZE), 1.0)

    with pytest.raises(ValueError, match="at most 4 frames, and this pass has 6"):
        fake_engine._decode(encoded)

    assert fake_engine._decoder.frames == []
