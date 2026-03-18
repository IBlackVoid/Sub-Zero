#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TARGET_TRIPLE="${1:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DIST_DIR="${ROOT_DIR}/dist"

mkdir -p "${DIST_DIR}"

if [[ -n "${TARGET_TRIPLE}" ]]; then
  cargo build --release --target "${TARGET_TRIPLE}"
  BIN_PATH="${ROOT_DIR}/target/${TARGET_TRIPLE}/release/sub-zero"
  if [[ ! -f "${BIN_PATH}" ]]; then
    BIN_PATH="${ROOT_DIR}/target/${TARGET_TRIPLE}/release/sub-zero.exe"
  fi
  PACKAGE_NAME="sub-zero_${TARGET_TRIPLE}_${STAMP}"
else
  cargo build --release
  BIN_PATH="${ROOT_DIR}/target/release/sub-zero"
  if [[ ! -f "${BIN_PATH}" ]]; then
    BIN_PATH="${ROOT_DIR}/target/release/sub-zero.exe"
  fi
  HOST="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
  PACKAGE_NAME="sub-zero_${HOST}_${STAMP}"
fi

if [[ ! -f "${BIN_PATH}" ]]; then
  echo "error: release binary not found at ${BIN_PATH}" >&2
  exit 1
fi

OUT_DIR="${DIST_DIR}/${PACKAGE_NAME}"
mkdir -p "${OUT_DIR}"

cp "${BIN_PATH}" "${OUT_DIR}/"
cp "${ROOT_DIR}/README.md" "${OUT_DIR}/"
cp "${ROOT_DIR}/scripts/release/bootstrap_models.sh" "${OUT_DIR}/"

tar -czf "${DIST_DIR}/${PACKAGE_NAME}.tar.gz" -C "${DIST_DIR}" "${PACKAGE_NAME}"

echo "release package created:"
echo "  ${DIST_DIR}/${PACKAGE_NAME}.tar.gz"
