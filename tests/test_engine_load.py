import dataclasses
import json
import logging
from pathlib import Path

import onnxruntime as ort
import pytest

from holos_tts import engine
from holos_tts.engine import HolosEngine, check_manifest


@dataclasses.dataclass
class Call:
    path: str
    providers: list[str]
    threads: int
    arena: bool
    precision: str | None
    spinning: str | None


def _config_entry(options: ort.SessionOptions, key: str) -> str | None:
    try:
        return options.get_session_config_entry(key)
    except RuntimeError:
        return None


class FakeSession:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return self._providers


class FakeInferenceSession:
    """Stands for ``onnxruntime.InferenceSession``: records the path and the options of each session."""

    def __init__(self) -> None:
        self.calls: list[Call] = []

    def __call__(self, path: str, options: ort.SessionOptions, providers: list[str]) -> FakeSession:
        self.calls.append(
            Call(
                path,
                providers,
                options.intra_op_num_threads,
                options.enable_cpu_mem_arena,
                _config_entry(options, engine.QUANT_PRECISION_ENTRY),
                _config_entry(options, engine.SPINNING_ENTRY),
            )
        )
        return FakeSession(providers)


def _manifest(**changes: object) -> dict[str, object]:
    manifest = {
        "format": "holos-onnx",
        "format_version": 2,
        "encoder": "encoder.onnx",
        "prosody": "prosody.onnx",
        "decoder": "decoder.onnx",
        "voices": "voices.npz",
        "vocab": list(engine.VOCAB),
        "sample_rate": 24000,
        "samples_per_frame": 600,
        "max_frames": 1000,
        "decoder_mode": "exact",
    }
    manifest.update(changes)
    return manifest


@dataclasses.dataclass
class Env:
    sessions: FakeInferenceSession
    hub: Path
    cache_dir: Path
    downloads: list[tuple[str, str]]

    def write_manifest(self, bundle: str, manifest: dict[str, object]) -> None:
        path = self.hub / bundle / engine.MANIFEST_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    sessions = FakeInferenceSession()
    hub = tmp_path / "hub"
    downloads: list[tuple[str, str]] = []

    def download(repo: str, filename: str, *, revision: str) -> str:
        assert repo == engine.MODEL_REPO
        downloads.append((filename, revision))
        return str(hub / filename)

    monkeypatch.setattr(ort, "InferenceSession", sessions)
    monkeypatch.setattr(ort, "preload_dlls", lambda: None)
    monkeypatch.setattr(engine, "hf_hub_download", download)
    # no flags line: the tests that need the CPU features write this file
    monkeypatch.setattr(engine, "CPUINFO", tmp_path / "cpuinfo")
    result = Env(sessions, hub, tmp_path / "cache", downloads)
    for bundle in engine.BUNDLES.values():
        result.write_manifest(bundle, _manifest())

    return result


def test_cpu_opens_the_three_stages_of_the_int8_bundle(env):
    HolosEngine({})

    assert [call.path for call in env.sessions.calls] == [
        str(env.hub / "onnx-int8" / name) for name in ("encoder.onnx", "prosody.onnx", "decoder.onnx")
    ]
    assert env.downloads[0] == ("onnx-int8/holos_onnx.json", engine.MODEL_REVISION)
    assert {revision for _name, revision in env.downloads} == {engine.MODEL_REVISION}
    assert all(call.providers == ["CPUExecutionProvider"] for call in env.sessions.calls)


def test_the_stage_files_come_from_the_manifest(env):
    env.write_manifest("onnx-int8", _manifest(decoder="decoder_v2.onnx"))

    HolosEngine({})

    assert env.sessions.calls[-1].path == str(env.hub / "onnx-int8" / "decoder_v2.onnx")


def test_the_frame_limit_comes_from_the_manifest(env):
    env.write_manifest("onnx-int8", _manifest(max_frames=640))

    assert HolosEngine({})._max_frames == 640


def test_cpu_sessions_turn_the_arena_and_the_thread_spinning_off(env):
    HolosEngine({}, threads=3)

    assert [(call.threads, call.arena, call.spinning) for call in env.sessions.calls] == [(3, False, "0")] * 3


def test_cuda_opens_the_fp32_bundle_with_the_cuda_provider_first(env):
    HolosEngine({}, device="cuda")

    assert [call.path for call in env.sessions.calls] == [
        str(env.hub / "onnx" / name) for name in ("encoder.onnx", "prosody.onnx", "decoder.onnx")
    ]
    assert all(call.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"] for call in env.sessions.calls)


def test_cuda_keeps_the_arena_and_the_thread_spinning(env):
    HolosEngine({}, device="cuda")

    assert [(call.arena, call.spinning) for call in env.sessions.calls] == [(True, None)] * 3


def test_cuda_without_the_cuda_provider_in_a_session_is_an_error(env, monkeypatch):
    monkeypatch.setattr(ort, "InferenceSession", lambda *_args, **_kwargs: FakeSession(["CPUExecutionProvider"]))

    with pytest.raises(RuntimeError, match=r"DEVICE=cuda.*CPUExecutionProvider.*encoder"):
        HolosEngine({}, device="cuda")


def test_the_old_graph_cache_is_removed(env, caplog):
    old = env.cache_dir / engine.OLD_CACHE_FOLDER
    old.mkdir(parents=True)
    (old / "holos_cpu_int8.0123456789abcdef.onnx").write_bytes(b"old optimized graph")
    other = env.cache_dir / "stanza-pretrain"
    other.mkdir()

    with caplog.at_level(logging.INFO, logger=engine.LOG.name):
        HolosEngine({}, cache_dir=env.cache_dir)

    assert not old.exists()
    assert other.is_dir()
    assert "Removing the optimized graphs of the older model" in caplog.text


def test_the_engine_writes_nothing_to_the_cache(env, caplog):
    env.cache_dir.mkdir()

    with caplog.at_level(logging.INFO, logger=engine.LOG.name):
        HolosEngine({}, cache_dir=env.cache_dir)

    assert list(env.cache_dir.iterdir()) == []
    assert "Removing" not in caplog.text


def test_a_bundle_that_the_server_cannot_run_is_refused(env):
    env.write_manifest("onnx-int8", _manifest(format_version=3))

    with pytest.raises(ValueError, match="format"):
        HolosEngine({})

    assert env.sessions.calls == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format": "other"}, r"has the format \('other', 2\)"),
        ({"format_version": 1}, r"has the format \('holos-onnx', 1\)"),
        ({"decoder_mode": "window"}, "decoder mode 'window'"),
        ({"decoder_mode": None}, "decoder mode None"),
        ({"vocab": list(engine.VOCAB[:-1])}, "vocabulary"),
        ({"vocab": [*engine.VOCAB[:-1], "x"]}, "vocabulary"),
    ],
)
def test_check_manifest_refuses_what_the_engine_cannot_run(changes, message):
    with pytest.raises(ValueError, match=message):
        check_manifest(_manifest(**changes))


def test_check_manifest_accepts_the_bundle_format():
    check_manifest(_manifest())


@pytest.fixture
def cpuinfo(tmp_path, monkeypatch):
    """The file with the CPU features. It does not exist until a test writes it."""
    path = tmp_path / "cpuinfo"
    monkeypatch.setattr(engine, "CPUINFO", path)
    monkeypatch.setattr(engine.platform, "processor", lambda: "test-cpu")
    monkeypatch.setattr(engine.platform, "machine", lambda: "test-arch")
    return path


def test_cpu_flags_come_from_the_flags_line_of_proc_cpuinfo(cpuinfo):
    cpuinfo.write_text(
        "processor\t: 0\nmodel name\t: Test CPU\nflags\t\t: fpu sse2 avx2\nbogomips\t: 1.0\n", encoding="utf-8"
    )

    assert engine._cpu_flags() == "flags\t\t: fpu sse2 avx2"


def test_cpu_flags_fall_back_when_proc_cpuinfo_has_no_flags_line(cpuinfo):
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Test CPU\n", encoding="utf-8")

    assert engine._cpu_flags() == "test-cputest-arch"


def test_cpu_flags_fall_back_without_proc_cpuinfo(cpuinfo):
    assert not cpuinfo.exists()
    assert engine._cpu_flags() == "test-cputest-arch"


X86_FLAGS = "fpu sse2 sse4_1 avx fma avx2"


def _write_flags(path: Path, flags: str) -> None:
    path.write_text(f"processor\t: 0\nmodel name\t: Test CPU\nflags\t\t: {flags}\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (X86_FLAGS, True),
        (f"{X86_FLAGS} avx_vnni", False),
        (f"{X86_FLAGS} avx512f avx512_vnni", False),
        # a longer flag that only contains the name is not VNNI
        (f"{X86_FLAGS} avx_vnni_int8x", True),
    ],
)
def test_quant_precision_follows_the_vnni_flags(cpuinfo, flags, expected):
    _write_flags(cpuinfo, flags)

    assert engine._needs_quant_precision() is expected


def test_quant_precision_is_off_without_a_flags_line(cpuinfo):
    cpuinfo.write_text("processor\t: 0\nFeatures\t: fp asimd\n", encoding="utf-8")

    assert engine._needs_quant_precision() is False


def test_quant_precision_is_off_without_proc_cpuinfo(cpuinfo):
    assert engine._needs_quant_precision() is False


def test_cpu_without_vnni_opens_each_stage_with_the_exact_int8_matmul(env, caplog):
    _write_flags(engine.CPUINFO, X86_FLAGS)

    with caplog.at_level(logging.INFO, logger=engine.LOG.name):
        HolosEngine({})

    assert [call.precision for call in env.sessions.calls] == ["1"] * 3
    assert "The CPU has no VNNI" in caplog.text


def test_cpu_with_vnni_keeps_the_fast_int8_matmul(env):
    _write_flags(engine.CPUINFO, f"{X86_FLAGS} avx_vnni")

    HolosEngine({})

    assert [call.precision for call in env.sessions.calls] == [None] * 3


def test_cuda_never_sets_the_exact_int8_matmul(env):
    _write_flags(engine.CPUINFO, X86_FLAGS)

    HolosEngine({}, device="cuda")

    assert [call.precision for call in env.sessions.calls] == [None] * 3
