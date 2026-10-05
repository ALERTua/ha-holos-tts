---
name: containers
description: Read this file before you build or run an image, or change the Dockerfile, .dockerignore or the container recipes of the justfile.
metadata:
  type: project
---

- The Dockerfile has two final targets: `cpu`, the last stage and the default, and `cuda`.
- `.dockerignore` is a list of the allowed files. Each file that the Dockerfile copies must be on that list. `pyproject.toml` names `README.md` as the readme, so the build needs it. The droast hook reports a missing file as DF077.
- The CUDA image has two CUDA stacks. ONNX Runtime loads its own CUDA libraries. CTranslate2 needs the CUDA 12 cuBLAS of the `nvidia-cublas-cu12` package, which `LD_LIBRARY_PATH` of the `cuda` stage points to.
- Inside the container, the servers keep the ports 8000 and 10200. `docker-compose.yml` and the justfile map `HTTP_PORT` and `WYOMING_PORT` to the host.
- The justfile runs the containers with wslc, the container command of WSL. wslc has no compose. A `-e` value overrides the same variable from `--env-file`.
- The justfile mounts the folder `data` next to it as `/data`. A model load from a Windows folder is about 1.7 times slower than from a named volume. This does not apply to a Linux host.
- In Git Bash, set `MSYS_NO_PATHCONV=1` for each wslc command with a container path, or Git Bash changes `/data` into a Windows path.
