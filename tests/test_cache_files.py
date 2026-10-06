import os
from pathlib import Path
from typing import Never

import pytest

from holos_tts import cache_files

OLD = b"old content"
NEW = b"new content"


class WriteError(Exception):
    pass


def write_then_raise(target: Path, error: type[BaseException], *, folder: bool = False) -> Never:
    with cache_files.atomic_path(target) as temp:
        if folder:
            temp.mkdir()
            (temp / "new.txt").write_bytes(NEW)
        else:
            temp.write_bytes(NEW)

        raise error


def test_file_target_gets_the_content_and_leaves_no_temp(tmp_path):
    target = tmp_path / "deep" / "folder" / "graph.onnx"

    with cache_files.atomic_path(target) as temp:
        assert not target.exists()
        temp.write_bytes(NEW)

    assert target.read_bytes() == NEW
    assert list(target.parent.iterdir()) == [target]


def test_temp_is_next_to_the_target_and_has_the_pid_in_its_name(tmp_path):
    target = tmp_path / "graph.onnx"

    with cache_files.atomic_path(target) as temp:
        assert temp == tmp_path / f"graph.onnx.{os.getpid()}.partial"
        assert not temp.exists()
        temp.write_bytes(NEW)


def test_folder_target_gets_the_files_and_leaves_no_temp(tmp_path):
    target = tmp_path / "cache" / "copy"

    with cache_files.atomic_path(target) as temp:
        temp.mkdir()
        (temp / "a.txt").write_bytes(NEW)

    assert [file.name for file in target.iterdir()] == ["a.txt"]
    assert (target / "a.txt").read_bytes() == NEW
    assert list(target.parent.iterdir()) == [target]


def test_existing_file_target_is_replaced(tmp_path):
    target = tmp_path / "graph.onnx"
    target.write_bytes(OLD)

    with cache_files.atomic_path(target) as temp:
        temp.write_bytes(NEW)

    assert target.read_bytes() == NEW


def test_existing_folder_target_is_replaced_without_its_old_files(tmp_path):
    target = tmp_path / "copy"
    target.mkdir()
    (target / "old.txt").write_bytes(OLD)

    with cache_files.atomic_path(target) as temp:
        temp.mkdir()
        (temp / "new.txt").write_bytes(NEW)

    assert [file.name for file in target.iterdir()] == ["new.txt"]


@pytest.mark.parametrize("error", [WriteError, KeyboardInterrupt])
def test_error_removes_the_temp_file_keeps_the_old_target_and_raises(tmp_path, error):
    target = tmp_path / "graph.onnx"
    target.write_bytes(OLD)

    with pytest.raises(error):
        write_then_raise(target, error)

    assert target.read_bytes() == OLD
    assert list(tmp_path.iterdir()) == [target]


def test_error_removes_the_temp_folder_keeps_the_old_target_and_raises(tmp_path):
    target = tmp_path / "copy"
    target.mkdir()
    (target / "old.txt").write_bytes(OLD)

    with pytest.raises(WriteError):
        write_then_raise(target, WriteError, folder=True)

    assert [file.name for file in target.iterdir()] == ["old.txt"]
    assert list(tmp_path.iterdir()) == [target]


def test_error_without_an_old_target_leaves_nothing(tmp_path):
    with pytest.raises(WriteError):
        write_then_raise(tmp_path / "copy", WriteError, folder=True)

    assert list(tmp_path.iterdir()) == []


def test_a_temp_that_the_caller_did_not_write_raises_and_leaves_nothing(tmp_path):
    target = tmp_path / "graph.onnx"

    with pytest.raises(FileNotFoundError), cache_files.atomic_path(target):
        pass

    assert list(tmp_path.iterdir()) == []


def test_failed_rename_removes_the_temp_and_raises(tmp_path, monkeypatch):
    target = tmp_path / "graph.onnx"

    def refuse(self, destination) -> Never:
        msg = "read-only"
        raise PermissionError(msg)

    def write() -> None:
        with cache_files.atomic_path(target) as temp:
            temp.write_bytes(NEW)

    monkeypatch.setattr(Path, "replace", refuse)

    with pytest.raises(PermissionError):
        write()

    assert list(tmp_path.iterdir()) == []


def test_leftovers_of_this_target_go_before_the_write_and_other_targets_stay(tmp_path):
    target = tmp_path / "graph.onnx"
    leftover_file = tmp_path / f"graph.onnx.{os.getpid() + 1}.partial"
    leftover_folder = tmp_path / "graph.onnx.4242.partial"
    other_target = tmp_path / "graph.onnx.5.partial.bak"
    other_name = tmp_path / "other.onnx.4242.partial"
    leftover_file.write_bytes(OLD)
    leftover_folder.mkdir()
    (leftover_folder / "half.bin").write_bytes(OLD)
    other_target.write_bytes(OLD)
    other_name.write_bytes(OLD)

    with cache_files.atomic_path(target) as temp:
        assert not leftover_file.exists()
        assert not leftover_folder.exists()
        temp.write_bytes(NEW)

    assert sorted(tmp_path.iterdir()) == sorted([target, other_target, other_name])


def test_partials_match_only_the_temp_paths_of_the_target(tmp_path):
    target = tmp_path / "a[1].onnx"
    mine = [tmp_path / "a[1].onnx.7.partial", tmp_path / "a[1].onnx.8.partial"]
    others = [tmp_path / "a1.onnx.7.partial", tmp_path / "a[1].onnx", tmp_path / "a[1].onnx.7.tmp"]
    for path in (*mine, *others):
        path.write_bytes(OLD)

    assert sorted(cache_files.partials(target)) == sorted(mine)


def test_partials_of_a_missing_folder_are_empty(tmp_path):
    assert cache_files.partials(tmp_path / "missing" / "graph.onnx") == []


def test_remove_path_removes_a_file_and_a_folder_with_files(tmp_path):
    file = tmp_path / "a.bin"
    folder = tmp_path / "copy"
    file.write_bytes(OLD)
    folder.mkdir()
    (folder / "b.bin").write_bytes(OLD)

    cache_files.remove_path(file)
    cache_files.remove_path(folder)

    assert list(tmp_path.iterdir()) == []


def test_remove_path_ignores_a_missing_path(tmp_path):
    cache_files.remove_path(tmp_path / "missing")


def test_remove_path_removes_a_link_to_a_folder_and_keeps_the_files_of_the_folder(tmp_path):
    folder = tmp_path / "real"
    link = tmp_path / "link"
    folder.mkdir()
    (folder / "b.bin").write_bytes(OLD)
    try:
        link.symlink_to(folder, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow a symbolic link")

    cache_files.remove_path(link)

    assert not link.is_symlink()
    assert (folder / "b.bin").read_bytes() == OLD
