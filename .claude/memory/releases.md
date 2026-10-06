---
name: releases
description: Read this file before you make a release, change the image tags, or tell a user which image tag to run.
metadata:
  type: project
---

- The image tags on `ghcr.io/alertua/ha-holos-tts`: `edge` is the newest commit of `main`. `latest`, `1`, `1.2` and `1.2.3` come only from a release tag `v1.2.3`. Each tag also exists with the suffix `-cuda`. The shared workflow `docker-build.yml` of `ALERTua/ci-workflows` makes these tags.
- A pull request to `main` runs the tests and builds both images, and pushes nothing.
- The Wyoming Info reports the version of the installed package, not the image tag. Before a release, change `version` in `pyproject.toml`, run `uv lock`, and merge that change by a pull request. Then put an annotated tag `vX.Y.Z` on that commit of `main`, and create a GitHub Release with the changes since the last tag.
- The tag `v1.0.0` kept the package version 0.1.0, so its Wyoming Info said 0.1.0.
