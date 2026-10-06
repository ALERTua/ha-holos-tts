---
name: containers
description: Read this file before you build or run an image, or change the Dockerfile, .dockerignore or the container recipes of the justfile.
metadata:
  type: project
---

- The Dockerfile has two final targets: `cpu`, the last stage and the default, and `cuda`.
- `.dockerignore` is a list of the allowed files. Each file that the Dockerfile copies must be on that list. `pyproject.toml` names `README.md` as the readme, so the build needs it. The droast hook reports a missing file as DF077.
- The final stage copies the venv of the stage `deps-cpu` or `deps-cuda`, which has the dependencies without the project. Then `uv pip install` puts the project wheel from `locked/` into its own small layer. A change of `src` or `README.md` thus does not change the layer of the dependencies (2.5 GB in `cuda`), and a push does not send it again. Do not copy the venv of a stage that also has the project.
- A new run of `uv sync` in `deps-cpu` or `deps-cuda` must give the same files, for example after a new uv release or a change of `pyproject.toml`. Then BuildKit takes the pushed layer of `.venv/lib` from its cache by content. For that, uv writes no installer metadata, and one `compileall -f --invalidation-mode unchecked-hash` process writes all bytecode. Do not turn on `UV_COMPILE_BYTECODE` in these stages: its bytecode carries the install time. `pyvenv.cfg` names the uv version, so it goes into the second, small layer of the venv.
- To check that the layer of the packages stays the same, build with a `docker-container` builder of the BuildKit version of alert-server. `wslc build` uses the `docker` driver of buildx: it reuses the layer only for the same uv version, and it ignores `rewrite-timestamp`.
- The CUDA image has two CUDA stacks. ONNX Runtime loads its own CUDA libraries. CTranslate2 needs the CUDA 12 cuBLAS of the `nvidia-cublas-cu12` package, which `LD_LIBRARY_PATH` of the `cuda` stage points to.
- The server starts without a network when `/data` already has the models, because each download has a pinned revision. To make sure of it, run the image with `wslc run --network none`, and request speech from inside the container with `wslc exec`.
- Inside the container, the servers keep the ports 8000 and 10200. `docker-compose.yml` and the justfile map `HTTP_PORT` and `WYOMING_PORT` to the host.
- The justfile runs the containers with wslc, the container command of WSL. wslc has no compose. A `-e` value overrides the same variable from `--env-file`.
- The justfile mounts the folder `data` next to it as `/data`. A model load from a Windows folder is about 1.7 times slower than from a named volume. This does not apply to a Linux host.
- In Git Bash, set `MSYS_NO_PATHCONV=1` for each wslc command with a container path, or Git Bash changes `/data` into a Windows path.
