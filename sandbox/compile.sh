#!/bin/sh
# Re-lock in a throwaway uv container (digest-pinned).
set -e
cd "$(dirname "$0")"
docker run --rm -v "$PWD":/w -w /w ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 \
  uv pip compile requirements.in --generate-hashes --python-version 3.12 \
  --python-platform aarch64-unknown-linux-gnu --index-strategy unsafe-best-match "$@" -o requirements.lock
