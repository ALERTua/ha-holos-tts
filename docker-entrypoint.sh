#!/usr/bin/env bash
# Start command of the image.
# Without UV_OVERRIDE, it runs the command at once with the packages of the image in $APP_DIR/.venv.
# With UV_OVERRIDE, it installs the locked requirements of the image with the overrides into $DATA_DIR/venv,
# and runs the command from that venv. uv reads UV_EXTRA_INDEX_URL and the other UV_* variables by itself.
set -euo pipefail

if [[ -n "${UV_OVERRIDE:-}" && ! -f "${UV_OVERRIDE}" ]]; then
  echo "WARNING: UV_OVERRIDE file not found: ${UV_OVERRIDE}. The server starts with the packages of the image." >&2
  unset UV_OVERRIDE
fi

if [[ -n "${UV_OVERRIDE:-}" ]]; then
  image_venv="${APP_DIR}/.venv"
  venv="${DATA_DIR}/venv"
  echo "UV_OVERRIDE is ${UV_OVERRIDE}. Installing the packages into ${venv}." >&2

  if [[ ! -x "${venv}/bin/python" ]]; then
    uv venv --allow-existing --python /usr/local/bin/python3 "${venv}"
  fi

  # The locked versions come from two indexes: PyPI and the PyTorch CPU index.
  # With the exact versions of the lock, uv must look for each version on all indexes.
  # wheels.txt replaces the git URLs inside the wheels from git, because the image has no git.
  # uv skips the packages that are already installed with the right version, so a restart installs nothing.
  uv pip install \
    --python "${venv}" \
    --index-strategy "${UV_INDEX_STRATEGY:-unsafe-best-match}" \
    --requirements "${APP_DIR}/locked/requirements.txt" \
    --overrides "${APP_DIR}/locked/wheels.txt" \
    --overrides "${UV_OVERRIDE}"

  export VIRTUAL_ENV="${venv}"
  export PATH="${venv}/bin:${PATH}"
  # CTranslate2 finds the CUDA libraries through LD_LIBRARY_PATH, so it must point into the venv that runs.
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//"${image_venv}/"/"${venv}/"}"
  fi
fi

exec "$@"
