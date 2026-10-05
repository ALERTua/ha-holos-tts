# Two images from one file:
#   docker build -t holos-tts .                    CPU image (the default, the last stage)
#   docker build -t holos-tts:cuda --target cuda . NVIDIA GPU image
ARG PYTHON_VERSION=3.12
# empty: the newest uv; a version such as 0.12.20: that uv
ARG UV_VERSION=
ARG UV_IMAGE=ghcr.io/astral-sh/uv:${UV_VERSION:+${UV_VERSION}-}python${PYTHON_VERSION}-trixie-slim
ARG APP_DIR=/app
ARG UV_CACHE_DIR=/uv-cache


# -----------------------------------------------------------------
FROM ${UV_IMAGE} AS builder

ARG APP_DIR
ARG UV_CACHE_DIR

ENV \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    UV_NO_PROGRESS=true \
    UV_NO_DEV=true \
    UV_CACHE_DIR=$UV_CACHE_DIR \
    UV_PROJECT_ENVIRONMENT=$APP_DIR/.venv

WORKDIR $APP_DIR

# the stress model and the phonemizer install from git
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------
FROM builder AS builder-cpu

RUN --mount=type=cache,target=$UV_CACHE_DIR \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --extra cpu

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync --no-editable --extra cpu

# The locked requirements for docker-entrypoint.sh, which installs them into $DATA_DIR/venv when UV_OVERRIDE is set.
# The runtime image has no git, so the git dependencies and the project go there as wheels.
# A wheel from git can require another git package by its URL, so wheels.txt also overrides these URLs.
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv export --quiet --extra cpu --no-emit-project --no-hashes --no-header --no-annotate --emit-index-url --output-file /tmp/locked.txt \
    && { grep ' @ git+' /tmp/locked.txt || true; } > /tmp/git.txt \
    && if [ -s /tmp/git.txt ]; then python -m pip wheel --quiet --disable-pip-version-check --no-deps --wheel-dir locked -r /tmp/git.txt; fi \
    && uv build --wheel --out-dir locked \
    && for wheel in locked/*.whl; do name="${wheel##*/}"; echo "${name%%-*} @ file://$PWD/$wheel"; done > locked/wheels.txt \
    && { grep -v ' @ git+' /tmp/locked.txt; cat locked/wheels.txt; } > locked/requirements.txt


# -----------------------------------------------------------------
FROM builder AS builder-cuda

RUN --mount=type=cache,target=$UV_CACHE_DIR \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --extra cuda

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync --no-editable --extra cuda

# The same locked requirements as in builder-cpu, for the cuda extra.
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv export --quiet --extra cuda --no-emit-project --no-hashes --no-header --no-annotate --emit-index-url --output-file /tmp/locked.txt \
    && { grep ' @ git+' /tmp/locked.txt || true; } > /tmp/git.txt \
    && if [ -s /tmp/git.txt ]; then python -m pip wheel --quiet --disable-pip-version-check --no-deps --wheel-dir locked -r /tmp/git.txt; fi \
    && uv build --wheel --out-dir locked \
    && for wheel in locked/*.whl; do name="${wheel##*/}"; echo "${name%%-*} @ file://$PWD/$wheel"; done > locked/wheels.txt \
    && { grep -v ' @ git+' /tmp/locked.txt; cat locked/wheels.txt; } > locked/requirements.txt


# -----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-trixie AS runtime

ARG APP_DIR
ARG PUID=1000
ARG PGID=1000

LABEL maintainer="ALERT <alexey.rubasheff@gmail.com>"
LABEL org.opencontainers.image.description="Ukrainian HolosTTS speech server for Home Assistant: Wyoming and OpenAI-compatible APIs"

ENV \
    PATH=$APP_DIR/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    HOME=/tmp \
    APP_DIR=$APP_DIR \
    DATA_DIR=/data \
    HF_HOME=/data/huggingface \
    STANZA_RESOURCES_DIR=/data/stanza \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HTTP_PORT=8000 \
    WYOMING_PORT=10200 \
    UV_CACHE_DIR=/data/uv_cache \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_PROGRESS=1

RUN groupadd --gid ${PGID} appuser \
    && useradd --uid ${PUID} --gid ${PGID} --no-log-init --no-create-home appuser \
    && mkdir -p /data \
    && chown ${PUID}:${PGID} /data

# uv installs the packages of UV_OVERRIDE at start
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

WORKDIR $APP_DIR

EXPOSE 8000 10200

VOLUME /data

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "-m", "holos_tts.healthcheck"]

USER ${PUID}:${PGID}

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["holos-tts"]


# -----------------------------------------------------------------
FROM runtime AS cuda

ARG APP_DIR
ARG PYTHON_VERSION

# CTranslate2 of the verbalizer finds the CUDA 12 cuBLAS of the pip packages through this path.
# ONNX Runtime loads its own CUDA libraries by itself.
# docker-entrypoint.sh moves these paths to $DATA_DIR/venv when it runs the server from there.
ENV \
    DEVICE=cuda \
    LD_LIBRARY_PATH=$APP_DIR/.venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cublas/lib:$APP_DIR/.venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_runtime/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=builder-cuda $APP_DIR/.venv $APP_DIR/.venv
COPY --from=builder-cuda $APP_DIR/locked $APP_DIR/locked


# -----------------------------------------------------------------
FROM runtime AS cpu

ARG APP_DIR

ENV DEVICE=cpu

COPY --from=builder-cpu $APP_DIR/.venv $APP_DIR/.venv
COPY --from=builder-cpu $APP_DIR/locked $APP_DIR/locked
