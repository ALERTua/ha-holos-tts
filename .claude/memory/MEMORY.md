# Project memory index: ha-holos-tts

Read this index from the top. A line with no link is the full rule.

- Run Python only through `uv run`, from the repository root.
- The server runs on Linux in a container, and the tests also run on Windows. Give each `/proc` or `/sys` path a module constant that a test can point to a temporary file, for example `config.CGROUP_CPU_MAX` and `engine.CPUINFO`.
- Do not change production code to make a test pass. Put the difference into the test code.
- Fix a type error instead of a `# ty: ignore` comment. An ignore that stays must give its reason on the same line.
- After a change of the engine, the verbalizer or the text pipeline, transcribe the speech with an STT model and compare the words. A waveform comparison does not show broken speech. The STT model drops most numbers, so judge the result by the words.
- [Models and pins](models.md) — before you change a model, a model revision, the verbalizer, the stress model or the text pipeline, read this file.
- [Worker process](worker.md) — before you change the worker process, the lock, the warm-up, the idle unload or a cancel path, read this file.
- [Caches](caches.md) — before you change a file under `/data/cache` or the code that writes it, read this file.
- [Memory and speed](memory-and-speed.md) — before you measure or change the memory or the speed of the server, read this file.
- [Containers](containers.md) — before you build or run an image, or change the Dockerfile, `.dockerignore` or the container recipes of the justfile, read this file.
- [Tooling](tooling.md) — before you change pre-commit, CI, ty, coverage or the pytest configuration, read this file.
- [Home Assistant](home-assistant.md) — before you test the server with Home Assistant, read this file.
- [Releases](releases.md) — before you make a release, change the image tags, or tell a user which image tag to run, read this file.
