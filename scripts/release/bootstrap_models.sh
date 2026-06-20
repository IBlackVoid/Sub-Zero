#!/usr/bin/env bash
#
# VoiDex release bootstrap: download Whisper + NLLB weights from
# HuggingFace at *pinned commit SHAs*, then materialise them at the
# directory layout the engine expects.
#
# Why pinned SHAs and not just model names:
#   - Reproducibility: a `0.1.0` build downloads byte-identical weights
#     today and a year from now, regardless of upstream re-uploads.
#   - Supply chain: a compromised tag (e.g. an attacker pushing a new
#     `main`) cannot retroactively change the bits a VoiDex user is
#     running. The SHA is the trust anchor; nothing else is.
#
# Updating: run `git ls-remote https://huggingface.co/<org>/<model> HEAD`
# to print the current SHA, paste it below, commit. Do not skip the
# `git ls-remote` step — picking a "fresh" SHA by hand defeats the
# point of pinning.

set -euo pipefail

# ─── pinned HuggingFace revisions ──────────────────────────────────────
# Last refreshed: 2026-05-22 via `git ls-remote ... HEAD`.

NLLB_REPO="facebook/nllb-200-distilled-600M"
NLLB_SHA="f8d333a098d19b4fd9a8b18f94170487ad3f821d"

WHISPER_BASE_REPO="openai/whisper-base"
WHISPER_BASE_SHA="e37978b90ca9030d5170a5c07aadb050351a65bb"

WHISPER_LARGE_REPO="openai/whisper-large-v3"
WHISPER_LARGE_SHA="06f233fe06e710322aca913c1bc4249a0d71fce1"

# ─── derived paths ─────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VOIDEX_HOME="${VOIDEX_HOME:-$HOME/.voidex}"
MODEL_ROOT="${VOIDEX_HOME}/models"

mkdir -p "${MODEL_ROOT}/whisper" "${MODEL_ROOT}/nllb"

echo "voidex bootstrap:"
echo "  model_root=${MODEL_ROOT}"

# ─── path A: pre-staged local models (offline / air-gapped) ────────────
# A maintainer who has already downloaded the weights into ./models can
# skip every network call. We copy and exit. No SHA verification is
# performed in this path; the operator owns provenance.

if [[ -d "${ROOT_DIR}/models" ]] && [[ "${VOIDEX_BOOTSTRAP_OFFLINE:-0}" = "1" ]]; then
  echo "  source_models=${ROOT_DIR}/models (offline mode)"
  shopt -s nullglob
  for item in "${ROOT_DIR}"/models/*; do
    base="$(basename "${item}")"
    case "${base}" in
      whisper*|ggml*|*.pt|*.bin)
        cp -R "${item}" "${MODEL_ROOT}/whisper/" 2>/dev/null || true
        ;;
      nllb*|*distilled*|*ct2*)
        cp -R "${item}" "${MODEL_ROOT}/nllb/" 2>/dev/null || true
        ;;
    esac
  done
  shopt -u nullglob
  echo "  done (offline)."
  exit 0
fi

# ─── path B: HuggingFace download at pinned SHA ────────────────────────
# Prefer the official `huggingface-cli` because it handles LFS, ETag
# caching, and resumable downloads correctly. Fall back to `git clone`
# only when the maintainer has no Python.

download_repo() {
  local repo="$1"
  local sha="$2"
  local dest="$3"

  echo "  fetching ${repo}@${sha:0:12}..."

  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download \
      --quiet \
      --revision "${sha}" \
      --local-dir "${dest}" \
      "${repo}"
  elif command -v git >/dev/null 2>&1; then
    # git-LFS must be present or weight files will be tiny pointer stubs.
    if ! git lfs version >/dev/null 2>&1; then
      echo "error: git lfs is required for the git fallback. install it from https://git-lfs.com" >&2
      exit 2
    fi
    rm -rf "${dest}"
    git clone --quiet "https://huggingface.co/${repo}" "${dest}"
    (cd "${dest}" && git checkout --quiet "${sha}")
  else
    echo "error: need either huggingface-cli or git+git-lfs to bootstrap models" >&2
    echo "       install with: pip install -U 'huggingface_hub[cli]'" >&2
    exit 2
  fi

  # Belt-and-braces: confirm the directory's current HEAD matches the
  # pin. huggingface-cli writes a ref but git might not, so we only
  # assert when a `.git` is present.
  if [[ -d "${dest}/.git" ]]; then
    local got
    got="$(git -C "${dest}" rev-parse HEAD)"
    if [[ "${got}" != "${sha}" ]]; then
      echo "error: ${repo} HEAD=${got} but pin was ${sha}" >&2
      exit 3
    fi
  fi
}

download_repo "${NLLB_REPO}"          "${NLLB_SHA}"          "${MODEL_ROOT}/nllb/nllb-200-distilled-600M"
download_repo "${WHISPER_BASE_REPO}"  "${WHISPER_BASE_SHA}"  "${MODEL_ROOT}/whisper/whisper-base"
download_repo "${WHISPER_LARGE_REPO}" "${WHISPER_LARGE_SHA}" "${MODEL_ROOT}/whisper/whisper-large-v3"

echo "  done."
echo "  whisper_dir=${MODEL_ROOT}/whisper"
echo "  nllb_dir=${MODEL_ROOT}/nllb"
echo "  pins:"
echo "    nllb-200-distilled-600M ${NLLB_SHA}"
echo "    whisper-base            ${WHISPER_BASE_SHA}"
echo "    whisper-large-v3        ${WHISPER_LARGE_SHA}"
