"""
Write an int8 copy of a float32 CTranslate2 model with numpy.

The copy has the layout of ``ModelSpec._serialize``, so the loader skips its own quantization.
"""

from __future__ import annotations

import shutil
import struct
from math import prod
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

from .cache_files import atomic_path

if TYPE_CHECKING:
    from pathlib import Path

MODEL_FILE = "model.bin"
BINARY_VERSION = 6
# data type ids follow the DataType enum of CTranslate2: float32, int8, int16, int32, float16, bfloat16
FLOAT32_ID = 0
INT8_ID = 1
ITEM_SIZES = (4, 1, 2, 4, 2, 2)
INT8_MAX = np.float32(127)
# bytes of float32 rows that the converter reads at one time
ROW_BLOCK_BYTES = 16 * 2**20
COPY_CHUNK_BYTES = 16 * 2**20


class ModelFormatError(ValueError):
    """The model file has a layout that the converter does not know."""


def convert_to_int8(source_dir: Path, target_dir: Path) -> None:
    """Write ``target_dir`` as a copy of the model in ``source_dir`` with int8 weights."""
    with atomic_path(target_dir) as work_dir:
        work_dir.mkdir()
        with (source_dir / MODEL_FILE).open("rb") as source, (work_dir / MODEL_FILE).open("wb") as target:
            _convert(source, target)

        for path in source_dir.iterdir():
            if path.name == MODEL_FILE:
                continue

            if path.is_dir():
                shutil.copytree(path, work_dir / path.name)
            else:
                shutil.copyfile(path, work_dir / path.name)


def is_quantizable(name: str) -> bool:
    """Whether the loader quantizes the variable: ``Model::is_quantizable`` takes each name that ends with weight."""
    return name.endswith("weight")


def row_shape(name: str, dims: tuple[int, ...]) -> tuple[int, int]:
    """Rows and row length of a weight, as the loader splits it into rows with one scale each."""
    if len(dims) == 3 and "conv" in name:  # noqa: PLR2004 - a convolution weight is (out, in, kernel)
        return dims[0], dims[1] * dims[2]

    depth = dims[-1]
    return prod(dims) // depth, depth


def quantize_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Int8 values and the float32 scale of each row, by the formula of CTranslate2."""
    amax = np.abs(rows).max(axis=1)
    amax[amax == 0] = INT8_MAX
    scale = INT8_MAX / amax
    return np.rint(rows * scale[:, np.newaxis]).astype(np.int8), scale.astype(np.float32)


def _read(source: BinaryIO, size: int) -> bytes:
    data = source.read(size)
    if len(data) != size:
        msg = "The model file ends too early"
        raise ModelFormatError(msg)

    return data


def _read_struct(source: BinaryIO, fmt: str) -> tuple[int, ...]:
    return struct.unpack(fmt, _read(source, struct.calcsize(fmt)))


def _read_string(source: BinaryIO) -> str:
    (size,) = _read_struct(source, "<H")
    # the stored size counts the closing NUL byte
    return _read(source, size)[:-1].decode("utf-8")


def _write_string(target: BinaryIO, text: str) -> None:
    data = text.encode("utf-8")
    target.write(struct.pack("<H", len(data) + 1))
    target.write(data + b"\0")


def _write_header(target: BinaryIO, name: str, dims: tuple[int, ...], *, dtype_id: int) -> None:
    _write_string(target, name)
    target.write(struct.pack("<B", len(dims)))
    target.write(struct.pack(f"<{len(dims)}I", *dims))
    target.write(struct.pack("<BI", dtype_id, prod(dims) * ITEM_SIZES[dtype_id]))


def _convert(source: BinaryIO, target: BinaryIO) -> None:
    (version,) = _read_struct(source, "<I")
    if version != BINARY_VERSION:
        msg = f"The model file has binary version {version}, but the converter reads only version {BINARY_VERSION}"
        raise ModelFormatError(msg)

    target.write(struct.pack("<I", version))
    _write_string(target, _read_string(source))
    revision, count = _read_struct(source, "<II")
    # the variable count comes before the variables, so a first pass finds the weights that get a scale
    variables = []
    for _ in range(count):
        name, dims, dtype_id = _read_variable_header(source)
        variables.append((name, dims, dtype_id, source.tell()))
        source.seek(prod(dims) * ITEM_SIZES[dtype_id], 1)

    scales = sum(_needs_quantization(name, dims, dtype_id) for name, dims, dtype_id, _ in variables)
    target.write(struct.pack("<II", revision, count + scales))
    for name, dims, dtype_id, offset in variables:
        source.seek(offset)
        if _needs_quantization(name, dims, dtype_id):
            _write_quantized(source, target, name=name, dims=dims)
        else:
            _write_header(target, name, dims, dtype_id=dtype_id)
            _copy(source, target, prod(dims) * ITEM_SIZES[dtype_id])

    # the alias table follows the last variable
    shutil.copyfileobj(source, target, COPY_CHUNK_BYTES)


def _read_variable_header(source: BinaryIO) -> tuple[str, tuple[int, ...], int]:
    name = _read_string(source)
    (rank,) = _read_struct(source, "<B")
    dims = _read_struct(source, f"<{rank}I")
    dtype_id, size = _read_struct(source, "<BI")
    if dtype_id >= len(ITEM_SIZES):
        msg = f"Variable {name!r} has the unknown data type id {dtype_id}"
        raise ModelFormatError(msg)

    if size != prod(dims) * ITEM_SIZES[dtype_id]:
        msg = f"Variable {name!r} has {size} bytes, but its shape {dims} needs another size"
        raise ModelFormatError(msg)

    return name, dims, dtype_id


def _needs_quantization(name: str, dims: tuple[int, ...], dtype_id: int) -> bool:
    if not is_quantizable(name) or dtype_id == INT8_ID:
        return False

    if dtype_id != FLOAT32_ID or not dims:
        msg = f"Weight {name!r} with shape {dims} and data type id {dtype_id} is not a float32 tensor"
        raise ModelFormatError(msg)

    return True


def _write_quantized(source: BinaryIO, target: BinaryIO, *, name: str, dims: tuple[int, ...]) -> None:
    rows, depth = row_shape(name, dims)
    row_bytes = depth * ITEM_SIZES[FLOAT32_ID]
    block = max(1, ROW_BLOCK_BYTES // row_bytes)
    scales = np.empty(rows, dtype=np.float32)
    _write_header(target, name, dims, dtype_id=INT8_ID)
    for start in range(0, rows, block):
        count = min(block, rows - start)
        values = np.frombuffer(_read(source, count * row_bytes), dtype="<f4").reshape(count, depth)
        quantized, block_scales = quantize_rows(values)
        scales[start : start + count] = block_scales
        target.write(quantized.tobytes())

    _write_header(target, f"{name}_scale", (rows,), dtype_id=FLOAT32_ID)
    target.write(scales.astype("<f4").tobytes())


def _copy(source: BinaryIO, target: BinaryIO, size: int) -> None:
    while size:
        chunk = _read(source, min(size, COPY_CHUNK_BYTES))
        target.write(chunk)
        size -= len(chunk)
