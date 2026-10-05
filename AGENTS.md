# AGENTS.md

Guidance for AI coding agents in this repository. Make small changes, make sure that each claim agrees with the code,
and obey the boundaries at the end.

## Project memory

The traps and the decisions with their reasons are in the project memory. Before you start a task, read this
file: [.claude/memory/MEMORY.md](.claude/memory/MEMORY.md).

## Project overview

ha-holos-tts is a Ukrainian text-to-speech server for Home Assistant. It runs
the [HolosTTS](https://huggingface.co/patriotyk/HolosTTS) model of Serhiy Stetskovych (patriotyk) in one container, and
gives two interfaces:

- a Wyoming server (port 10200) for the Wyoming integration of Home Assistant;
- an OpenAI-compatible speech API (port 8000) for the openai_tts integration and Music Assistant.

Before the speech, the text goes through three steps: a verbalizer writes numbers and symbols as words, a stress model
puts the stress marks, and a phonemizer makes IPA phonemes. See `README.md` for the user view: the configuration, the
API and the measured memory and speed.

## Repository layout

| Path                                                           | What it holds                                                                  |
|----------------------------------------------------------------|--------------------------------------------------------------------------------|
| `src/holos_tts/__main__.py`                                    | starts the HTTP API and the Wyoming server in one process                      |
| `src/holos_tts/config.py`                                      | `Settings`, read from environment variables                                    |
| `src/holos_tts/synthesizer.py`                                 | the main-process side of the worker: lock, warm-up, idle unload, cancel safety |
| `src/holos_tts/worker.py`, `worker_entry.py`                   | the worker process that holds the three models                                 |
| `src/holos_tts/text.py`                                        | sentence split, stress marks of the user, the verbalizer filter                |
| `src/holos_tts/frontend.py`                                    | stress model (stanza) and IPA                                                  |
| `src/holos_tts/verbalizer.py`, `ct2_int8.py`                   | the CTranslate2 verbalizer and its int8 copy                                   |
| `src/holos_tts/engine.py`                                      | HolosTTS in ONNX Runtime, the voices, the optimized-graph cache                |
| `src/holos_tts/stanza_cache.py`                                | fast copy of the stanza word vectors                                           |
| `src/holos_tts/cache_files.py`                                 | the atomic write of the cache files                                            |
| `src/holos_tts/memory.py`                                      | glibc heap settings for the model loads                                        |
| `src/holos_tts/openai_api.py`, `wyoming_server.py`, `audio.py` | the two interfaces and the audio formats                                       |
| `tests/`                                                       | pytest suite; `fake_worker.py` holds the fake worker processes                 |
| `Dockerfile`                                                   | targets `cpu` (default) and `cuda`                                             |
| `justfile`                                                     | the task runner                                                                |

## Commands

The project uses [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). Run each command from the
repository root. Run Python only through `uv run`.

- `just install`: after a clone, and after a change of the dependencies.
- `just lint`: after each code change. It formats, fixes the lint findings and runs ty.
- `just test`: after each change.
- `just cov`: when you add or change tests. CI fails when the total coverage is below 85 %.
- `just pre`: before you finish a change. It runs all pre-commit hooks.
- `just build`, `just run`, `just stop`, `just logs`, `just health`, `just say "текст"`: to see the server work in a
  local container. `just run` reads `.env` and mounts the folder `data` as `/data`. `just help` lists all recipes.

## Code style

- Python 3.12 (`requires-python = "==3.12.*"`). The Dockerfile uses the same version.
- Ruff with `select = ["ALL"]` and line length 120. Prefer a narrow `# noqa: RULE - reason` over a new global ignore.
- ty checks `src` only. Fix a type error instead of an ignore comment.
- A docstring has one to three lines, and a comment has one line that states a constraint. Write only English in the
  code, the comments and the commit messages.
- A comment, a docstring or a name never refers to a plan, an issue number or a review round. It states the rule itself.
- The main process never imports torch, stanza, ONNX Runtime or CTranslate2. Import them inside the functions of the
  worker modules (`# noqa: PLC0415`).

## Testing

- The tests run on Windows and on Linux. They use fake worker processes, not the real models.
- Do not change production code to make a test pass. Put the difference into the test code.
- A spawned fake worker must live in `tests/fake_worker.py`. A target in a test module makes the child process import
  the whole test module.
- Prove a new test with a mutation: break the line that the test protects, and make sure that the test fails.

## Agent boundaries

- Ask the user before you change an invariant of the project memory, a model, a model revision or a default value.
- Never run a git command that changes the repository (`add`, `commit`, `push`, `reset`, `checkout`, `stash`, `restore`)
  unless the user asks for it.
- Do not change a production Home Assistant.
