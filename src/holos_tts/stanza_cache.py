"""Fast copy of the stanza pretrain vectors: a word list and a ``.npy`` file load faster than the ``.pt`` file."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from stanza.models.common.pretrain import Pretrain, PretrainedWordVocab

from .cache_files import atomic_path

if TYPE_CHECKING:
    from collections.abc import Callable

LOG = logging.getLogger(__name__)

_FOLDER = "stanza-pretrain"
_VOCAB_FILE = "vocab.txt"
_EMB_FILE = "emb.npy"
_META_FILE = "meta.json"
_META_KEYS = ("lang", "idx", "cutoff", "lower")


def install(cache_dir: Path) -> None:
    """Make ``Pretrain.load`` read and write the fast copy in ``cache_dir``. A second call changes only the folder."""
    original = getattr(Pretrain.load, "__wrapped__", Pretrain.load)
    Pretrain.load = _wrap(original, cache_dir)


def _wrap(original: Callable[[Pretrain], None], cache_dir: Path) -> Callable[[Pretrain], None]:
    @functools.wraps(original)
    def load(self: Pretrain) -> None:
        if self.filename is None or not Path(self.filename).is_file():
            original(self)
            return

        root = cache_dir / _FOLDER
        folder = _folder(root, Path(self.filename))
        if folder.is_dir():
            try:
                _read(self, folder)
            except Exception:
                LOG.warning(
                    "Cannot read the fast copy %s of the pretrain vectors, so the server removes it",
                    folder,
                    exc_info=True,
                )
                shutil.rmtree(folder, ignore_errors=True)
            else:
                return

        original(self)
        _write(self, root)

    return load


def _folder(root: Path, source: Path) -> Path:
    """Return the folder of the copy of ``source``: the copy depends on the path, the size and the time of the file."""
    stat = source.stat()
    parts = (str(source.resolve()), str(stat.st_size), str(stat.st_mtime_ns))
    return root / hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _read(pretrain: Pretrain, folder: Path) -> None:
    meta = json.loads((folder / _META_FILE).read_text(encoding="utf-8"))
    words = (folder / _VOCAB_FILE).read_bytes().decode("utf-8").split("\n")
    emb = np.load(folder / _EMB_FILE)
    if emb.ndim != 2 or not len(words) == meta["words"] == emb.shape[0]:  # noqa: PLR2004 - a matrix has two axes
        msg = f"the word count is {len(words)}, the meta says {meta['words']}, the matrix has the shape {emb.shape}"
        raise ValueError(msg)

    state = {key: meta[key] for key in _META_KEYS}
    state["_unit2id"] = {word: index for index, word in enumerate(words)}
    state["_id2unit"] = words
    pretrain._vocab = PretrainedWordVocab.load_state_dict(state)  # noqa: SLF001
    pretrain._emb = torch.from_numpy(emb)  # noqa: SLF001


def _remove_stale_copies(root: Path, keep: Path, source: str) -> None:
    """Remove the other copies in ``root`` that name the same ``source`` file in their meta file."""
    with contextlib.suppress(OSError):
        for other in root.iterdir():
            if other == keep or not other.is_dir():
                continue

            try:
                meta = json.loads((other / _META_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue

            if isinstance(meta, dict) and meta.get("source") == source:
                shutil.rmtree(other, ignore_errors=True)


def _write(pretrain: Pretrain, root: Path) -> None:
    """Save the loaded vectors of ``pretrain`` as the fast copy. A failed save only logs a warning."""
    try:
        vocab = pretrain._vocab  # noqa: SLF001
        words: list[str] = vocab._id2unit  # noqa: SLF001
        emb = pretrain._emb.numpy()  # noqa: SLF001
        # the word list file has one word per line, and the copy gives each word its position as the id
        if not words or any(vocab._unit2id.get(word) != index or "\n" in word for index, word in enumerate(words)):  # noqa: SLF001
            LOG.debug("The pretrain vectors do not fit the fast copy, so the server does not save it")
            return

        if emb.ndim != 2 or emb.shape[0] != len(words):  # noqa: PLR2004
            LOG.debug("The pretrain matrix does not fit the word list, so the server does not save the fast copy")
            return

        started = time.monotonic()
        source = Path(pretrain.filename)
        meta: dict[str, Any] = {key: getattr(vocab, key) for key in _META_KEYS}
        meta["words"] = len(words)
        meta["source"] = str(source.resolve())
        folder = _folder(root, source)
        with atomic_path(folder) as temp:
            # the temp folders of other copies, and the ``<pid>.tmp`` folders of the old naming
            for leftover in (*root.glob("*.partial"), *root.glob("*.tmp")):
                shutil.rmtree(leftover, ignore_errors=True)

            temp.mkdir()
            (temp / _VOCAB_FILE).write_bytes("\n".join(words).encode("utf-8"))
            np.save(temp / _EMB_FILE, emb)
            (temp / _META_FILE).write_text(json.dumps(meta), encoding="utf-8")

        LOG.info("Wrote the fast copy of the pretrain vectors to %s in %.1f s", folder, time.monotonic() - started)
        _remove_stale_copies(root, folder, meta["source"])
    except Exception:
        LOG.warning("Cannot save the fast copy of the pretrain vectors to %s", root, exc_info=True)
