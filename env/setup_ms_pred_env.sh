#!/usr/bin/env bash
# Reproduce the vendor/ms-pred Python environment exactly.
#
# vendor/ms-pred is a git submodule pinned to a specific upstream commit
# (coleygroup/ms-pred). We do NOT commit build artifacts (uv.lock, .venv)
# inside that submodule's own working tree -- we don't have push access to
# the upstream repo, so a commit made there would only exist locally and
# would break `git submodule update` for anyone else who clones this repo.
#
# Instead, the resolved lock file is tracked here, in the superproject, and
# copied into place before `uv sync`. This reproduces the exact dependency
# versions we tested with, while keeping vendor/ms-pred an untouched copy
# of upstream.
#
# Usage (from the repo root, after `git submodule update --init --recursive`):
#   ./env/setup_ms_pred_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MS_PRED_DIR="$REPO_ROOT/vendor/ms-pred"

if [ ! -f "$MS_PRED_DIR/pyproject.toml" ]; then
    echo "vendor/ms-pred is missing or not initialized." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

cp "$REPO_ROOT/env/ms-pred-uv.lock" "$MS_PRED_DIR/uv.lock"

cd "$MS_PRED_DIR"
uv sync --extra cu124 --extra test --frozen

echo ""
echo "Environment ready at vendor/ms-pred/.venv"
echo "Invoke scripts as: vendor/ms-pred/.venv/bin/python src/ms_pred/..."
