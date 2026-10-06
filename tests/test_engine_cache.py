import dataclasses
import logging
import os
import weakref
from pathlib import Path

import onnxruntime as ort
import pytest

from holos_tts import engine
from holos_tts.engine import HolosEngine

MODEL_FILE = "tiny_cpu.onnx"
STEM = "tiny_cpu"
OPTIMIZED = b"optimized graph"
BROKEN = b"broken graph"


@dataclasses.dataclass
class Call:
    path: str
    level: ort.GraphOptimizationLevel
    save_to: str
    providers: list[str]
    threads: int
    arena: bool
    precision: str | None


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
    """Stands for ``onnxruntime.InferenceSession``: records each call, writes the optimized graph, can fail."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.unloadable: set[str] = set()
        self.unloadable_once: set[str] = set()
        self.unsavable = False
        self.created: list[weakref.ref[FakeSession]] = []

    def __call__(self, path: str, options: ort.SessionOptions, providers: list[str]) -> FakeSession:
        save_to = options.optimized_model_filepath
        self.calls.append(
            Call(
                path,
                options.graph_optimization_level,
                save_to,
                providers,
                options.intra_op_num_threads,
                options.enable_cpu_mem_arena,
                _config_entry(options, engine.QUANT_PRECISION_ENTRY),
            )
        )
        if path in self.unloadable or path in self.unloadable_once:
            self.unloadable_once.discard(path)
            msg = f"cannot load {path}"
            raise ort.capi.onnxruntime_pybind11_state.InvalidArgument(msg)

        if save_to:
            Path(save_to).write_bytes(OPTIMIZED)
            if self.unsavable:
                msg = f"cannot save {save_to}"
                raise ort.capi.onnxruntime_pybind11_state.InvalidArgument(msg)

        session = FakeSession(providers)
        self.created.append(weakref.ref(session))
        return session


@dataclasses.dataclass
class Env:
    sessions: FakeInferenceSession
    model: Path
    cache_dir: Path

    @property
    def onnx_dir(self) -> Path:
        return self.cache_dir / engine.ONNX_CACHE_FOLDER

    @property
    def cached(self) -> Path:
        return self.onnx_dir / f"{STEM}.{engine._cache_key(MODEL_FILE, ort.__version__)}.onnx"


@pytest.fixture
def env(tmp_path, monkeypatch):
    model = tmp_path / "hub" / MODEL_FILE
    model.parent.mkdir()
    model.write_bytes(b"original graph")
    sessions = FakeInferenceSession()
    monkeypatch.setattr(ort, "InferenceSession", sessions)
    monkeypatch.setattr(ort, "preload_dlls", lambda: None)
    monkeypatch.setattr(ort, "__version__", "0.0.1")
    monkeypatch.setattr(engine, "MODEL_REVISION", "test-revision")
    monkeypatch.setattr(engine, "_FAILED_SAVES", set())
    # no flags line: the tests that need the CPU features write this file
    monkeypatch.setattr(engine, "CPUINFO", tmp_path / "cpuinfo")
    monkeypatch.setitem(engine.ONNX_FILES, "cpu", MODEL_FILE)
    monkeypatch.setitem(engine.ONNX_FILES, "cuda", "tiny_cuda.onnx")
    monkeypatch.setattr(engine, "hf_hub_download", lambda *_args, **_kwargs: str(model))
    return Env(sessions, model, tmp_path / "cache")


def test_miss_saves_the_optimized_graph_and_renames_it(env):
    HolosEngine({}, cache_dir=env.cache_dir)

    save, _load = env.sessions.calls
    assert save.path == str(env.model)
    assert save.level == ort.SessionOptions().graph_optimization_level
    assert Path(save.save_to) == env.onnx_dir / f"{env.cached.name}.{os.getpid()}.partial"
    assert env.cached.read_bytes() == OPTIMIZED
    assert list(env.onnx_dir.iterdir()) == [env.cached]


def test_miss_opens_the_saved_file_and_the_engine_uses_that_session(env):
    engine_ = HolosEngine({}, cache_dir=env.cache_dir)

    save, load = env.sessions.calls
    assert save.save_to
    assert load.path == str(env.cached)
    assert load.level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert not load.save_to
    assert engine_._session is env.sessions.created[-1]()
    assert env.sessions.created[0]() is None


def test_miss_drops_the_saving_session(env):
    engine_ = HolosEngine({}, cache_dir=env.cache_dir)

    saving, loaded = env.sessions.created
    assert saving() is None
    assert loaded() is engine_._session


def test_miss_keeps_the_thread_count_in_both_sessions(env):
    HolosEngine({}, threads=3, cache_dir=env.cache_dir)

    assert [call.threads for call in env.sessions.calls] == [3, 3]


def test_hit_loads_the_cache_without_optimization(env):
    env.onnx_dir.mkdir(parents=True)
    env.cached.write_bytes(OPTIMIZED)

    HolosEngine({}, cache_dir=env.cache_dir)

    (call,) = env.sessions.calls
    assert call.path == str(env.cached)
    assert call.level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert not call.save_to


def test_hit_keeps_the_thread_count(env):
    env.onnx_dir.mkdir(parents=True)
    env.cached.write_bytes(OPTIMIZED)

    HolosEngine({}, threads=3, cache_dir=env.cache_dir)

    assert env.sessions.calls[0].threads == 3


def test_broken_cache_falls_back_and_is_saved_again(env, caplog):
    env.onnx_dir.mkdir(parents=True)
    env.cached.write_bytes(BROKEN)
    env.sessions.unloadable_once.add(str(env.cached))

    with caplog.at_level(logging.WARNING, logger=engine.LOG.name):
        engine_ = HolosEngine({}, cache_dir=env.cache_dir)

    assert [call.path for call in env.sessions.calls] == [str(env.cached), str(env.model), str(env.cached)]
    assert env.cached.read_bytes() == OPTIMIZED
    assert engine_._session is env.sessions.created[-1]()
    assert "Cannot load the cached graph" in caplog.text


def test_failed_reopen_of_the_saved_file_deletes_it_and_opens_the_original_graph(env, caplog):
    env.sessions.unloadable.add(str(env.cached))

    with caplog.at_level(logging.WARNING, logger=engine.LOG.name):
        engine_ = HolosEngine({}, cache_dir=env.cache_dir)

    assert [(call.path, bool(call.save_to)) for call in env.sessions.calls] == [
        (str(env.model), True),
        (str(env.cached), False),
        (str(env.model), False),
    ]
    assert engine_._session is env.sessions.created[-1]()
    assert not env.cached.exists()
    assert list(env.onnx_dir.iterdir()) == []
    assert "Cannot load the saved graph" in caplog.text


def test_broken_cache_is_deleted_when_the_new_save_fails(env):
    env.onnx_dir.mkdir(parents=True)
    env.cached.write_bytes(BROKEN)
    env.sessions.unloadable.add(str(env.cached))
    env.sessions.unsavable = True

    engine_ = HolosEngine({}, cache_dir=env.cache_dir)

    assert [(call.path, bool(call.save_to)) for call in env.sessions.calls] == [
        (str(env.cached), False),
        (str(env.model), True),
        (str(env.model), False),
    ]
    assert engine_._session is env.sessions.created[-1]()
    assert not env.cached.exists()


def test_cuda_does_not_touch_the_cache(env):
    HolosEngine({}, device="cuda", cache_dir=env.cache_dir)

    (call,) = env.sessions.calls
    assert call.path == str(env.model)
    assert not call.save_to
    assert call.providers[0] == "CUDAExecutionProvider"
    assert not env.cache_dir.exists()


def test_cpu_without_a_cache_dir_opens_the_original_graph(env):
    HolosEngine({})

    (call,) = env.sessions.calls
    assert call.path == str(env.model)
    assert not call.save_to
    assert not env.cache_dir.exists()


def test_failed_save_keeps_the_session_and_removes_the_temp_file(env, caplog):
    env.sessions.unsavable = True

    with caplog.at_level(logging.WARNING, logger=engine.LOG.name):
        engine_ = HolosEngine({}, cache_dir=env.cache_dir)

    assert [(call.path, bool(call.save_to)) for call in env.sessions.calls] == [
        (str(env.model), True),
        (str(env.model), False),
    ]
    assert engine_._session is env.sessions.created[-1]()
    assert list(env.onnx_dir.iterdir()) == []
    assert f"Cannot open the original graph {env.model} to save its optimized copy to {env.cached}" in caplog.text
    (record,) = caplog.records
    assert record.exc_info


def test_failed_save_is_remembered_and_logged_once(env, caplog):
    env.sessions.unsavable = True

    with caplog.at_level(logging.INFO, logger=engine.LOG.name):
        HolosEngine({}, cache_dir=env.cache_dir)
        env.sessions.calls.clear()
        engine_ = HolosEngine({}, cache_dir=env.cache_dir)

    assert [(call.path, bool(call.save_to)) for call in env.sessions.calls] == [(str(env.model), False)]
    assert engine_._session is env.sessions.created[-1]()
    assert [record.levelno for record in caplog.records if "Cannot open" in record.message] == [logging.WARNING]


def test_failed_save_is_remembered_only_for_its_cache_file(env, monkeypatch):
    env.sessions.unsavable = True
    HolosEngine({}, cache_dir=env.cache_dir)
    monkeypatch.setattr(ort, "__version__", "0.0.2")
    env.sessions.calls.clear()

    HolosEngine({}, cache_dir=env.cache_dir)

    assert [(call.path, bool(call.save_to)) for call in env.sessions.calls] == [
        (str(env.model), True),
        (str(env.model), False),
    ]


def test_save_logs_the_time_it_took(env, caplog):
    with caplog.at_level(logging.INFO, logger=engine.LOG.name):
        HolosEngine({}, cache_dir=env.cache_dir)

    assert f"Wrote the optimized graph to {env.cached} in " in caplog.text


def test_save_removes_the_temp_files_that_a_killed_process_left(env):
    env.onnx_dir.mkdir(parents=True)
    leftovers = [
        env.onnx_dir / f"{env.cached.name}.{os.getpid() + 1}.partial",
        env.onnx_dir / f"{STEM}.0123456789abcdef.onnx.4242.partial",
        env.onnx_dir / f"{env.cached.name}.{os.getpid() + 1}.tmp",
        env.onnx_dir / f"{STEM}.0123456789abcdef.onnx.4242.tmp",
    ]
    other_model = env.onnx_dir / "other_model.0123456789abcdef.onnx.4242.partial"
    for file in (*leftovers, other_model):
        file.write_bytes(OPTIMIZED)

    HolosEngine({}, cache_dir=env.cache_dir)

    assert sorted(env.onnx_dir.iterdir()) == sorted([env.cached, other_model])


def test_cache_folder_that_cannot_be_created_keeps_the_session(env):
    env.cache_dir.mkdir()
    env.onnx_dir.write_bytes(b"a file where the folder must be")

    engine_ = HolosEngine({}, cache_dir=env.cache_dir)

    assert [call.path for call in env.sessions.calls] == [str(env.model)]
    assert not env.sessions.calls[0].save_to
    assert engine_._session is env.sessions.created[-1]()


def test_save_removes_the_graphs_of_other_keys(env):
    env.onnx_dir.mkdir(parents=True)
    stale = [env.onnx_dir / f"{STEM}.0123456789abcdef.onnx", env.onnx_dir / f"{STEM}.fedcba9876543210.onnx"]
    other_model = env.onnx_dir / "other_model.0123456789abcdef.onnx"
    for file in (*stale, other_model):
        file.write_bytes(OPTIMIZED)

    HolosEngine({}, cache_dir=env.cache_dir)

    assert sorted(env.onnx_dir.iterdir()) == sorted([env.cached, other_model])


def test_cpu_miss_turns_the_memory_arena_off(env):
    HolosEngine({}, cache_dir=env.cache_dir)

    assert [call.arena for call in env.sessions.calls] == [False, False]


def test_cpu_hit_turns_the_memory_arena_off(env):
    env.onnx_dir.mkdir(parents=True)
    env.cached.write_bytes(OPTIMIZED)

    HolosEngine({}, cache_dir=env.cache_dir)

    assert [call.arena for call in env.sessions.calls] == [False]


def test_cpu_fallback_after_a_failed_save_turns_the_memory_arena_off(env):
    env.sessions.unsavable = True

    HolosEngine({}, cache_dir=env.cache_dir)

    assert [(bool(call.save_to), call.arena) for call in env.sessions.calls] == [(True, False), (False, False)]


def test_cpu_after_a_broken_cache_turns_the_memory_arena_off(env):
    env.onnx_dir.mkdir(parents=True)
    env.cached.write_bytes(BROKEN)
    env.sessions.unloadable_once.add(str(env.cached))

    HolosEngine({}, cache_dir=env.cache_dir)

    assert [call.arena for call in env.sessions.calls] == [False, False, False]


def test_cpu_without_a_cache_dir_turns_the_memory_arena_off(env):
    HolosEngine({})

    assert [call.arena for call in env.sessions.calls] == [False]


def test_cuda_keeps_the_memory_arena(env):
    HolosEngine({}, device="cuda", cache_dir=env.cache_dir)

    assert [call.arena for call in env.sessions.calls] == [True]


def test_key_changes_with_the_onnx_runtime_version(env, monkeypatch):
    HolosEngine({}, cache_dir=env.cache_dir)
    first = env.cached
    monkeypatch.setattr(ort, "__version__", "0.0.2")

    HolosEngine({}, cache_dir=env.cache_dir)

    assert env.cached != first
    assert list(env.onnx_dir.iterdir()) == [env.cached]
    assert [bool(call.save_to) for call in env.sessions.calls] == [True, False, True, False]


def test_key_changes_with_the_cpu(monkeypatch):
    monkeypatch.setattr(engine, "_cpu_flags", lambda: "flags: sse2")
    first = engine._cache_key(MODEL_FILE, "0.0.1")
    monkeypatch.setattr(engine, "_cpu_flags", lambda: "flags: sse2 avx2")

    assert engine._cache_key(MODEL_FILE, "0.0.1") != first


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


def test_cpu_without_vnni_saves_and_loads_with_the_exact_int8_matmul(env):
    _write_flags(engine.CPUINFO, X86_FLAGS)

    HolosEngine({}, cache_dir=env.cache_dir)

    assert [call.precision for call in env.sessions.calls] == ["1", "1"]


def test_cpu_without_vnni_hit_loads_with_the_exact_int8_matmul(env):
    _write_flags(engine.CPUINFO, X86_FLAGS)
    env.onnx_dir.mkdir(parents=True)
    env.cached.write_bytes(OPTIMIZED)

    HolosEngine({}, cache_dir=env.cache_dir)

    assert [(call.path, call.precision) for call in env.sessions.calls] == [(str(env.cached), "1")]


def test_cpu_without_vnni_fallback_uses_the_exact_int8_matmul(env):
    _write_flags(engine.CPUINFO, X86_FLAGS)
    env.sessions.unsavable = True

    HolosEngine({}, cache_dir=env.cache_dir)

    assert [call.precision for call in env.sessions.calls] == ["1", "1"]


def test_cpu_with_vnni_keeps_the_fast_int8_matmul(env):
    _write_flags(engine.CPUINFO, f"{X86_FLAGS} avx_vnni")

    HolosEngine({}, cache_dir=env.cache_dir)

    assert [call.precision for call in env.sessions.calls] == [None, None]


def test_cuda_never_sets_the_exact_int8_matmul(env):
    _write_flags(engine.CPUINFO, X86_FLAGS)

    HolosEngine({}, device="cuda", cache_dir=env.cache_dir)

    assert [call.precision for call in env.sessions.calls] == [None]


def test_key_changes_when_the_exact_int8_matmul_turns_on(monkeypatch):
    monkeypatch.setattr(engine, "_cpu_flags", lambda: f"flags: {X86_FLAGS}")
    monkeypatch.setattr(engine, "_needs_quant_precision", lambda: False)
    without = engine._cache_key(MODEL_FILE, "0.0.1")
    monkeypatch.setattr(engine, "_needs_quant_precision", lambda: True)

    assert engine._cache_key(MODEL_FILE, "0.0.1") != without


def test_cuda_without_the_cuda_provider_in_the_session_is_an_error(env, monkeypatch):
    monkeypatch.setattr(ort, "InferenceSession", lambda *_args, **_kwargs: FakeSession(["CPUExecutionProvider"]))

    with pytest.raises(RuntimeError, match=r"DEVICE=cuda.*CPUExecutionProvider"):
        HolosEngine({}, device="cuda", cache_dir=env.cache_dir)
