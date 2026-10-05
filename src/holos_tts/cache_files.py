"""Write a cache file or folder to a temp path next to its target, then rename it onto the target."""

from __future__ import annotations

import contextlib
import glob
import os
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def partials(target: Path) -> list[Path]:
    """Return the temp paths of ``target`` that a running or a killed process wrote."""
    return list(target.parent.glob(f"{glob.escape(target.name)}.*.partial"))


def remove_path(path: Path) -> None:
    """Remove a file or a folder. An error does not raise."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


@contextlib.contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    """
    Yield a temp path next to ``target``. The caller writes a file or fills a folder there.

    A normal exit renames the temp path onto ``target``. An error removes the temp path and raises again.
    The pid in the name keeps the temp paths of two processes apart.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # a killed process leaves its temp path behind, and the pid of the next run differs
    for leftover in partials(target):
        remove_path(leftover)

    temp = target.with_name(f"{target.name}.{os.getpid()}.partial")
    try:
        yield temp
        if temp.is_dir():
            # a folder cannot replace a folder that has files, so the old folder goes first
            shutil.rmtree(target, ignore_errors=True)

        temp.replace(target)
    except BaseException:
        remove_path(temp)
        raise
