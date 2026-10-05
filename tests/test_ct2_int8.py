import os
import struct

import numpy as np
import pytest
from ctranslate2.specs import common_spec, model_spec

from holos_tts import cache_files, ct2_int8

DTYPES = (np.float32, np.int8, np.int16, np.int32, np.float16)


def write_string(f, text):
    data = text.encode()
    f.write(struct.pack("<H", len(data) + 1) + data + b"\0")


def write_model(path, variables, aliases, version=6):
    with path.open("wb") as f:
        f.write(struct.pack("<I", version))
        write_string(f, "FixtureSpec")
        f.write(struct.pack("<II", 3, len(variables)))
        for name, value in variables.items():
            write_string(f, name)
            f.write(struct.pack(f"<B{value.ndim}I", value.ndim, *value.shape))
            f.write(struct.pack("<BI", DTYPES.index(value.dtype.type), value.nbytes))
            f.write(value.tobytes())

        f.write(struct.pack("<I", len(aliases)))
        for alias, name in aliases.items():
            write_string(f, alias)
            write_string(f, name)


def read_model(path):
    data = path.read_bytes()
    pos = 0

    def take(fmt) -> tuple[int, ...]:
        nonlocal pos
        values = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return values

    def string() -> str:
        nonlocal pos
        (size,) = take("<H")
        pos += size
        return data[pos - size : pos - 1].decode()

    (version,) = take("<I")
    spec = string()
    revision, count = take("<II")
    variables = {}
    for _ in range(count):
        name = string()
        (rank,) = take("<B")
        dims = take(f"<{rank}I")
        dtype_id, size = take("<BI")
        variables[name] = np.frombuffer(
            data, DTYPES[dtype_id], size // np.dtype(DTYPES[dtype_id]).itemsize, pos
        ).reshape(dims)
        pos += size

    (alias_count,) = take("<I")
    aliases = {string(): string() for _ in range(alias_count)}
    assert pos == len(data)
    return version, spec, revision, variables, aliases


def reference_int8(value, rows, depth):
    # the formula of ModelSpec._quantize in the ctranslate2 package
    flat = value.reshape(rows, depth).astype(np.float32)
    amax = np.amax(np.absolute(flat), axis=1)
    amax[amax == 0] = 127.0
    scale = 127.0 / amax
    return np.rint(flat * np.expand_dims(scale, 1)).astype(np.int8).reshape(value.shape), scale


@pytest.fixture
def fixture_variables():
    rng = np.random.default_rng(41)
    linear = rng.normal(0, 0.37, (5, 6)).astype(np.float32)
    linear[2] = 0
    return {
        "encoder/linear_0/weight": linear,
        "encoder/linear_0/bias": rng.normal(0, 0.2, 5).astype(np.float32),
        "encoder/layer_norm/gamma": rng.normal(1, 0.1, 6).astype(np.float32),
        "encoder/layer_norm/beta": rng.normal(0, 0.1, 6).astype(np.float32),
        "encoder/conv_1/weight": rng.normal(0, 0.5, (4, 3, 2)).astype(np.float32),
        "decoder/mix/weight": rng.normal(0, 0.9, (2, 3, 4)).astype(np.float32),
        "decoder/embeddings/weight": rng.normal(0, 1.3, (9, 6)).astype(np.float32),
        "decoder/num_heads": np.array(5, dtype=np.int16),
        "decoder/scale_embeddings": np.array(2.5, dtype=np.float32),
    }


@pytest.fixture
def small_blocks(monkeypatch):
    # one or two rows for each block, so that each weight takes the block path many times
    monkeypatch.setattr(ct2_int8, "ROW_BLOCK_BYTES", 30)


@pytest.mark.usefixtures("small_blocks")
def test_converts_weights_and_keeps_the_rest(tmp_path, fixture_variables):
    source, target = tmp_path / "float", tmp_path / "out" / "int8"
    source.mkdir()
    aliases = {"decoder/projection/weight": "decoder/embeddings/weight"}
    write_model(source / "model.bin", fixture_variables, aliases)
    (source / "shared_vocabulary.json").write_text('["<s>", "x"]', encoding="utf-8")

    ct2_int8.convert_to_int8(source, target)

    version, spec, revision, variables, out_aliases = read_model(target / "model.bin")
    assert (version, spec, revision, out_aliases) == (6, "FixtureSpec", 3, aliases)
    assert (target / "shared_vocabulary.json").read_text(encoding="utf-8") == '["<s>", "x"]'
    assert cache_files.partials(target) == []
    rows = {
        "encoder/linear_0/weight": (5, 6),
        "encoder/conv_1/weight": (4, 6),
        "decoder/mix/weight": (6, 4),
        "decoder/embeddings/weight": (9, 6),
    }
    assert set(variables) == set(fixture_variables) | {f"{name}_scale" for name in rows}
    for name, (count, depth) in rows.items():
        expected, scale = reference_int8(fixture_variables[name], count, depth)
        assert variables[name].dtype == np.int8
        np.testing.assert_array_equal(variables[name], expected)
        assert variables[f"{name}_scale"].dtype == np.float32
        np.testing.assert_array_equal(variables[f"{name}_scale"], scale)

    assert variables["encoder/linear_0/weight_scale"][2] == 1
    for name in set(fixture_variables) - set(rows):
        assert variables[name].dtype == fixture_variables[name].dtype
        np.testing.assert_array_equal(variables[name], fixture_variables[name])


class FixtureModelSpec(model_spec.ModelSpec):
    def __init__(self, rng):
        super().__init__()
        self.embeddings = common_spec.EmbeddingsSpec()
        self.embeddings.weight = rng.normal(0, 0.8, (7, 6)).astype(np.float32)
        self.layer = [common_spec.LinearSpec() for _ in range(2)]
        self.layer[0].weight = rng.normal(0, 0.3, (5, 6)).astype(np.float32)
        self.layer[0].weight[3] = 0
        self.layer[0].bias = rng.normal(0, 0.1, 5).astype(np.float32)
        # the same values as the embeddings, so that the ctranslate2 writer stores an alias
        self.layer[1].weight = self.embeddings.weight.copy()
        self.conv = common_spec.Conv1DSpec()
        self.conv.weight = rng.normal(0, 0.6, (3, 2, 4)).astype(np.float32)
        self.norm = common_spec.LayerNormSpec()
        self.norm.gamma = rng.normal(1, 0.1, 6).astype(np.float32)
        self.norm.beta = rng.normal(0, 0.1, 6).astype(np.float32)
        self.heads = np.int16(3)

    @property
    def name(self):
        return "FixtureModelSpec"


def save_spec(path, quantization):
    spec = FixtureModelSpec(np.random.default_rng(17))
    spec.validate()
    spec.optimize(quantization)
    path.mkdir()
    spec.save(str(path))


@pytest.mark.parametrize("small", [False, True])
def test_writes_the_same_file_as_the_ctranslate2_converter(tmp_path, monkeypatch, small):
    if small:
        monkeypatch.setattr(ct2_int8, "ROW_BLOCK_BYTES", 30)

    save_spec(tmp_path / "float", None)
    save_spec(tmp_path / "expected", "int8")
    assert read_model(tmp_path / "float" / "model.bin")[4] == {"layer_1/weight": "embeddings/weight"}

    ct2_int8.convert_to_int8(tmp_path / "float", tmp_path / "int8")

    assert (tmp_path / "int8" / "model.bin").read_bytes() == (tmp_path / "expected" / "model.bin").read_bytes()


def test_refuses_an_unknown_binary_version(tmp_path, fixture_variables):
    source, target = tmp_path / "float", tmp_path / "int8"
    source.mkdir()
    write_model(source / "model.bin", fixture_variables, {}, version=5)

    with pytest.raises(ct2_int8.ModelFormatError, match="binary version 5"):
        ct2_int8.convert_to_int8(source, target)

    assert not target.exists()
    assert cache_files.partials(target) == []


def test_refuses_a_weight_that_is_not_float32(tmp_path, fixture_variables):
    source = tmp_path / "float"
    source.mkdir()
    fixture_variables["decoder/embeddings/weight"] = fixture_variables["decoder/embeddings/weight"].astype(np.float16)
    write_model(source / "model.bin", fixture_variables, {})

    with pytest.raises(ct2_int8.ModelFormatError, match="decoder/embeddings/weight"):
        ct2_int8.convert_to_int8(source, tmp_path / "int8")


def test_refuses_a_truncated_file(tmp_path, fixture_variables):
    source = tmp_path / "float"
    source.mkdir()
    write_model(source / "model.bin", fixture_variables, {})
    data = (source / "model.bin").read_bytes()
    (source / "model.bin").write_bytes(data[: len(data) // 2])

    with pytest.raises(ct2_int8.ModelFormatError):
        ct2_int8.convert_to_int8(source, tmp_path / "int8")


def test_removes_the_folder_of_a_crashed_conversion(tmp_path, fixture_variables):
    source, target = tmp_path / "float", tmp_path / "int8"
    source.mkdir()
    write_model(source / "model.bin", fixture_variables, {})
    leftover = tmp_path / f"{target.name}.{os.getpid() + 1}.partial"
    leftover.mkdir()
    (leftover / "model.bin").write_bytes(b"half")

    ct2_int8.convert_to_int8(source, target)

    assert not leftover.exists()
    assert read_model(target / "model.bin")[3]["decoder/embeddings/weight"].dtype == np.int8
