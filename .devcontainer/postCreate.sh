#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export POETRY_CACHE_DIR="${ROOT_DIR}/.cache/pypoetry"
mkdir -p "${POETRY_CACHE_DIR}"

if command -v poetry >/dev/null 2>&1; then
  poetry config virtualenvs.in-project true
  poetry install --only main --no-root
else
  echo "poetry not found; install dependencies manually"
fi
