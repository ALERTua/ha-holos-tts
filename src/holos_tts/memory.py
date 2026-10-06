"""
Heap settings of glibc for the worker.

A small mapping threshold while a model loads sends the freed load buffers back to the system. The threshold is small
only during a load, because torch runs about 70 % slower with it.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

M_MMAP_THRESHOLD = -3
# a buffer of this size or bigger gets its own mapping
LOAD_MMAP_THRESHOLD = 128 * 1024
RUN_MMAP_THRESHOLD = 32 * 1024 * 1024


def _libc() -> ctypes.CDLL | None:
    if not sys.platform.startswith("linux"):
        return None

    try:
        return ctypes.CDLL("libc.so.6")
    except OSError:
        return None


def set_mmap_threshold(size: int) -> None:
    """Allocate each heap buffer of ``size`` bytes or more with its own memory mapping."""
    libc = _libc()
    if libc is not None:
        with contextlib.suppress(AttributeError):
            libc.mallopt(M_MMAP_THRESHOLD, size)


def release_free_memory() -> None:
    """Give the free heap memory of this process back to the system."""
    gc.collect()
    libc = _libc()
    if libc is not None:
        with contextlib.suppress(AttributeError):
            libc.malloc_trim(0)


@contextlib.contextmanager
def model_loading() -> Iterator[None]:
    """Load a model with the small mapping threshold, then go back to the fast threshold and free the buffers."""
    set_mmap_threshold(LOAD_MMAP_THRESHOLD)
    try:
        yield
    finally:
        set_mmap_threshold(RUN_MMAP_THRESHOLD)
        release_free_memory()
