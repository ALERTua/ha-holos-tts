"""
Heap settings of glibc for the worker process.

A model load converts its weights in big temporary buffers. glibc keeps freed buffers in the heap of the process,
so after the stress model the verbalizer alone keeps about 1.5 GB instead of 0.5 GB. While a model loads, each big
buffer must get its own memory mapping, which goes back to the system when the buffer is freed.

The same setting makes torch on the CPU about 70 % slower while a model runs, because each tensor of a request then
gets a new mapping. So the worker uses the small threshold only while a model loads.

Other systems than Linux with glibc ignore these calls.
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
