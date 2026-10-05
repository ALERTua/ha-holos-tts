---
name: tooling
description: Read this file before you change pre-commit, CI, ty, coverage or the pytest configuration.
metadata:
  type: project
---

- ty is in the `dev` group, so `uv.lock` holds its version. The ty hook is a local hook: `uv run --locked ty check`. The ty-pre-commit hook runs `uv check`, which refuses the conflicting `cpu` and `cuda` extras. CI runs `uv run ty check`.
- `exclude-newer = "1 week"` in `[tool.uv]` makes uv skip releases younger than a week. An upgrade can therefore keep an older version than the newest one on PyPI.
- CI installs with `uv sync --locked --extra cpu`, so a `uv.lock` that does not agree with `pyproject.toml` stops CI.
- `[tool.ty.environment]` sets `python-platform = "linux"`, because the server runs on Linux. On Windows, ty types `multiprocessing.Pipe` as `PipeConnection` and reports errors that do not exist on Linux.
- `[tool.ty.src]` includes `src` only. The tests use fakes and wrong values on purpose.
- The droast hook builds droast with cargo on its first run.
- pre-commit has no pytest hook. CI runs the tests before it builds the images.
- The coverage gate is 85 %. Coverage does not see the code that runs in the spawned worker process.
- Pin each GitHub action by its commit SHA, with the version in a comment. Dependabot updates the SHA.
