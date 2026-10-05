import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Never

import pytest
import torch
from stanza.models.common.pretrain import Pretrain, PretrainedWordVocab

from holos_tts import stanza_cache

WORDS = ["слово", "ґанок", "abc"]
LANG = "xx"
IDX = 2
CUTOFF = 5


@dataclasses.dataclass
class Spy:
    """Stands in for the original ``Pretrain.load``: counts the calls and can refuse to run."""

    load: object
    calls: int = 0
    refuse: bool = False


@pytest.fixture
def spy(monkeypatch):
    original = Pretrain.load
    result = Spy(original)

    def load(self) -> None:
        result.calls += 1
        if result.refuse:
            msg = "the fast copy must answer"
            raise RuntimeError(msg)

        original(self)

    result.load = load
    monkeypatch.setattr(Pretrain, "load", load)
    return result


def make_file(path: Path, words: list[str] = WORDS, *, shift: float = 0.0) -> Path:
    pretrain = Pretrain(str(path))
    pretrain._vocab = PretrainedWordVocab(words, lang=LANG, idx=IDX, cutoff=CUTOFF, lower=True)
    pretrain._emb = torch.arange(len(pretrain._vocab) * 3, dtype=torch.float32).reshape(-1, 3) + shift
    pretrain.save(str(path))
    return path


def read(path: Path) -> tuple[dict, torch.Tensor]:
    pretrain = Pretrain(str(path))
    return dict(pretrain.vocab.state_dict()), pretrain.emb


def assert_same(actual: tuple[dict, torch.Tensor], expected: tuple[dict, torch.Tensor]) -> None:
    assert actual[0] == expected[0]
    assert actual[1].dtype == expected[1].dtype
    assert torch.equal(actual[1], expected[1])


def copies(cache_dir: Path) -> list[Path]:
    root = cache_dir / "stanza-pretrain"
    return sorted(root.iterdir()) if root.is_dir() else []


def test_first_load_saves_the_copy_and_returns_the_original_data(tmp_path, spy):
    file = make_file(tmp_path / "vectors.pt")
    expected = read(file)
    cache_dir = tmp_path / "cache"
    stanza_cache.install(cache_dir)

    actual = read(file)

    assert_same(actual, expected)
    (folder,) = copies(cache_dir)
    assert sorted(item.name for item in folder.iterdir()) == ["emb.npy", "meta.json", "vocab.txt"]
    assert (folder / "vocab.txt").read_text(encoding="utf-8").split("\n") == actual[0]["_id2unit"]
    assert spy.calls == 2


def test_second_load_comes_from_the_copy(tmp_path, spy):
    file = make_file(tmp_path / "vectors.pt")
    expected = read(file)
    stanza_cache.install(tmp_path / "cache")
    read(file)
    spy.refuse = True

    actual = read(file)

    assert_same(actual, expected)
    assert actual[0]["_id2unit"][-len(WORDS) :] == WORDS
    assert (actual[0]["lang"], actual[0]["idx"], actual[0]["cutoff"], actual[0]["lower"]) == (LANG, IDX, CUTOFF, True)


@pytest.mark.parametrize("spoil", ["emb.npy", "vocab.txt", "meta.json"])
def test_broken_copy_falls_back_and_is_saved_again(tmp_path, spy, caplog, spoil):
    file = make_file(tmp_path / "vectors.pt")
    expected = read(file)
    cache_dir = tmp_path / "cache"
    stanza_cache.install(cache_dir)
    read(file)
    (folder,) = copies(cache_dir)
    (folder / spoil).write_bytes(b"broken")
    spy.calls = 0

    with caplog.at_level(logging.WARNING, logger=stanza_cache.LOG.name):
        actual = read(file)

    assert_same(actual, expected)
    assert spy.calls == 1
    assert "Cannot read the fast copy" in caplog.text
    assert copies(cache_dir) == [folder]
    spy.refuse = True
    assert_same(read(file), expected)


def test_copy_with_another_word_count_is_not_used(tmp_path, spy):
    file = make_file(tmp_path / "vectors.pt")
    expected = read(file)
    cache_dir = tmp_path / "cache"
    stanza_cache.install(cache_dir)
    read(file)
    (folder,) = copies(cache_dir)
    (folder / "vocab.txt").write_bytes(b"\n".join(word.encode() for word in WORDS))
    spy.calls = 0

    assert_same(read(file), expected)

    assert spy.calls == 1


@pytest.mark.parametrize("words", [["один\nдва", "слово"], ["слово", "слово"]], ids=["newline", "repeated word"])
def test_vectors_that_do_not_fit_the_copy_are_not_saved(tmp_path, spy, caplog, words):
    file = make_file(tmp_path / "vectors.pt", words)
    expected = read(file)
    cache_dir = tmp_path / "cache"
    stanza_cache.install(cache_dir)

    with caplog.at_level(logging.DEBUG, logger=stanza_cache.LOG.name):
        assert_same(read(file), expected)
        assert_same(read(file), expected)

    assert copies(cache_dir) == []
    assert spy.calls == 3
    assert "do not fit the fast copy" in caplog.text


def test_failed_save_does_not_raise(tmp_path, spy, caplog):
    file = make_file(tmp_path / "vectors.pt")
    expected = read(file)
    cache_dir = tmp_path / "cache"
    cache_dir.write_bytes(b"a file where the folder must be")
    stanza_cache.install(cache_dir)

    with caplog.at_level(logging.WARNING, logger=stanza_cache.LOG.name):
        assert_same(read(file), expected)

    assert "Cannot save the fast copy" in caplog.text
    assert cache_dir.read_bytes() == b"a file where the folder must be"


def test_failed_replace_removes_the_temp_folder(tmp_path, spy, monkeypatch, caplog):
    file = make_file(tmp_path / "vectors.pt")
    expected = read(file)
    cache_dir = tmp_path / "cache"
    stanza_cache.install(cache_dir)

    def refuse(self, target) -> Never:
        msg = "read-only"
        raise PermissionError(msg)

    monkeypatch.setattr(Path, "replace", refuse)

    with caplog.at_level(logging.WARNING, logger=stanza_cache.LOG.name):
        assert_same(read(file), expected)

    assert "Cannot save the fast copy" in caplog.text
    assert list((cache_dir / "stanza-pretrain").iterdir()) == []


def test_changed_file_gets_a_new_copy_and_the_old_copy_of_the_same_source_goes(tmp_path, spy):
    new_words = [*WORDS, "ще"]
    expected = read(make_file(tmp_path / "other.pt", new_words, shift=0.5))
    cache_dir = tmp_path / "cache"
    stanza_cache.install(cache_dir)
    file = make_file(tmp_path / "vectors.pt")
    read(file)
    (old,) = copies(cache_dir)
    make_file(file, new_words, shift=0.5)

    assert_same(read(file), expected)

    (new,) = copies(cache_dir)
    assert new != old
    assert json.loads((new / "meta.json").read_text(encoding="utf-8"))["source"] == str(file.resolve())
    spy.refuse = True
    actual = read(file)
    assert_same(actual, expected)
    assert actual[0]["_id2unit"][-1] == "ще"


def test_copy_of_another_source_stays(tmp_path, spy):
    cache_dir = tmp_path / "cache"
    stanza_cache.install(cache_dir)
    file = make_file(tmp_path / "vectors.pt")
    other = make_file(tmp_path / "other.pt", [*WORDS, "ще"], shift=0.5)
    read(file)
    read(other)
    assert len(copies(cache_dir)) == 2
    other_copy = next(
        folder
        for folder in copies(cache_dir)
        if json.loads((folder / "meta.json").read_text(encoding="utf-8"))["source"] == str(other.resolve())
    )
    make_file(file, [*WORDS, "ще"], shift=0.25)

    read(file)

    assert len(copies(cache_dir)) == 2
    assert other_copy.is_dir()


def test_write_removes_the_temp_folders_that_a_killed_process_left(tmp_path, spy):
    file = make_file(tmp_path / "vectors.pt")
    cache_dir = tmp_path / "cache"
    root = cache_dir / "stanza-pretrain"
    leftovers = [
        root / f"0123456789abcdef.{os.getpid() + 1}.partial",
        root / f"fedcba9876543210.{os.getpid() + 1}.partial",
        root / f"{os.getpid() + 1}.tmp",
    ]
    for leftover in leftovers:
        leftover.mkdir(parents=True)
        (leftover / "emb.npy").write_bytes(b"half a file")

    stanza_cache.install(cache_dir)

    read(file)

    (folder,) = copies(cache_dir)
    assert folder.name not in {leftover.name for leftover in leftovers}


def test_write_logs_the_time_it_took(tmp_path, spy, caplog):
    file = make_file(tmp_path / "vectors.pt")
    stanza_cache.install(tmp_path / "cache")

    with caplog.at_level(logging.INFO, logger=stanza_cache.LOG.name):
        read(file)

    assert "Wrote the fast copy of the pretrain vectors to" in caplog.text


@pytest.mark.parametrize(
    "shape",
    [lambda emb: emb[:-1], lambda emb: emb[:, 0]],
    ids=["fewer rows than words", "one axis"],
)
def test_matrix_that_does_not_fit_the_word_list_is_not_saved(tmp_path, caplog, shape):
    pretrain = Pretrain(str(make_file(tmp_path / "vectors.pt")))
    pretrain._vocab = PretrainedWordVocab(WORDS, lang=LANG, idx=IDX, cutoff=CUTOFF, lower=True)
    pretrain._emb = torch.arange(len(pretrain._vocab) * 3, dtype=torch.float32).reshape(-1, 3)
    pretrain._emb = shape(pretrain._emb)
    root = tmp_path / "cache" / "stanza-pretrain"

    with caplog.at_level(logging.DEBUG, logger=stanza_cache.LOG.name):
        stanza_cache._write(pretrain, root)

    assert not root.exists()
    assert "does not fit the word list" in caplog.text


@pytest.mark.parametrize("name", [None, "missing.pt"])
def test_without_a_file_the_original_runs_unchanged(tmp_path, spy, name):
    cache_dir = tmp_path / "cache"
    stanza_cache.install(cache_dir)
    pretrain = Pretrain(None if name is None else str(tmp_path / name))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        pretrain.load()

    assert spy.calls == 1
    assert not cache_dir.exists()


def test_a_second_install_wraps_the_original_once(tmp_path, spy):
    file = make_file(tmp_path / "vectors.pt")
    first = tmp_path / "first"
    second = tmp_path / "second"
    stanza_cache.install(first)
    stanza_cache.install(second)

    read(file)

    assert Pretrain.load.__wrapped__ is spy.load
    assert spy.calls == 1
    assert copies(first) == []
    assert len(copies(second)) == 1
