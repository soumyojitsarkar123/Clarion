#!/usr/bin/env bash
# ==============================================================================
# Clarion Unified E2E Test Suite Runner Script
# Usage:
#   bash scripts/run_all_e2e_tests.sh
#   bash scripts/run_all_e2e_tests.sh --tier 2
#   bash scripts/run_all_e2e_tests.sh --tier all --verbose
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_PYTHON="${ROOT_DIR}/.venv_linux/bin/python"

if [[ ! -f "${VENV_PYTHON}" ]]; then
    echo "[ERROR] Linux virtual environment not found at ${VENV_PYTHON}" >&2
    exit 1
fi

export PYTHONPATH="${ROOT_DIR}/Clarion-Backend:${ROOT_DIR}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

exec "${VENV_PYTHON}" "${ROOT_DIR}/tests/e2e/run_e2e_suite.py" "$@"
