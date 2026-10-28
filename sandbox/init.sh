#!/usr/bin/env bash
# One-time, idempotent setup: .env, helper token, helper and relay binaries.
# Set LLM_HELPER_TOKEN to reuse the token of a helper that is already running.
set -euo pipefail
umask 077 # nothing secret is ever created group- or world-readable
cd "$(dirname "$0")"

(umask 022; mkdir -p ../.manual) # manual-mode prompts and the reply cache; the app mounts it writable

[ -f .env ] || cp .env.example .env
chmod 600 .env

# Append keys that .env.example gained since .env was created; existing values are never changed
[ -z "$(tail -c1 .env)" ] || echo >> .env
while IFS= read -r line; do
  case "$line" in *=*) ;; *) continue ;; esac
  case "$line" in '#'*) continue ;; esac
  grep -q "^${line%%=*}=" .env || echo "$line" >> .env
done < .env.example

if [ -n "${LLM_HELPER_TOKEN:-}" ]; then
  printf '%s\n' "$LLM_HELPER_TOKEN" > claude-helper/.token
elif [ ! -s claude-helper/.token ]; then
  head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > claude-helper/.token
fi
chmod 600 claude-helper/.token
tok=$(tr -d '[:space:]' < claude-helper/.token)

# Fill the token only if .env has none yet, or an override was given
env_tok=$(sed -n 's/^LLM_HELPER_TOKEN=//p' .env | head -n 1)
if [ -z "$env_tok" ] || [ -n "${LLM_HELPER_TOKEN:-}" ]; then
  { grep -v '^LLM_HELPER_TOKEN=' .env || true; echo "LLM_HELPER_TOKEN=$tok"; } > .env.tmp && mv .env.tmp .env
  env_tok=$tok
fi
chmod 600 .env
if [ "$env_tok" != "$tok" ]; then
  echo "WARNING: .env LLM_HELPER_TOKEN differs from claude-helper/.token, so the helper will answer 401." >&2
  echo "         Fix: LLM_HELPER_TOKEN=\$(cat claude-helper/.token) ./init.sh" >&2
fi

# Replace the placeholder Neo4j password only if .env still has it
if grep -q '^NEO4J_PASSWORD=changeme$' .env; then
  pw=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
  sed "s|^NEO4J_PASSWORD=changeme\$|NEO4J_PASSWORD=$pw|" .env > .env.tmp && mv .env.tmp .env
  chmod 600 .env
fi

(cd claude-helper && go build -o claude-helper . && chmod 755 claude-helper)
# Static linux binary for the scratch relay image; the non-root user in the image must be able to execute it
(cd relay && CGO_ENABLED=0 GOOS=linux GOARCH="$(go env GOARCH)" go build -o relay . && chmod 755 relay)
echo "ready: start the helper, then docker compose up -d mongo claude-relay (see README.md)"
