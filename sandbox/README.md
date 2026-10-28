# Local sandbox for the RAG tutorials

**Hard requirements: Apple Silicon Mac with Docker Desktop (the dependency lock targets aarch64 and the relay reaches the helper through `host.docker.internal`), and Go 1.26+.**

Runs modules 01 (naive RAG), 02 (metadata-filtered RAG), 03 (hybrid search), 04 (graph RAG) and 05 (agentic RAG) with no cloud accounts and no API keys. Everything runs in Docker; the only host tools are Docker Desktop, Go and Claude Code.

## Prerequisites

- Docker Desktop, running, on an Apple Silicon Mac.
- Go 1.26+ (builds the helper and relay).
- Claude Code installed and signed in with your own seat (`claude -p "hi"` must work).

## One-time setup

```sh
cd sandbox
./init.sh
```

It creates `.env` from `.env.example`, generates a helper token in `claude-helper/.token`, copies it into `.env`, sets a random Neo4j password, creates the `.manual` directory at the repo root, and builds the helper and relay. Re-running keeps an existing `.env` and token, and appends any key that `.env.example` has gained since (existing values are never changed). The one exception: `LLM_HELPER_TOKEN=<its token> ./init.sh` replaces the token in both, which is how you reuse the token of a helper that is already running.

## Start

```sh
cd sandbox
CLAUDE_MODEL=sonnet ./claude-helper/claude-helper &      # host helper, 127.0.0.1:8765
docker compose up -d mongo claude-relay                  # first run builds the app image
```

The helper listens on 8765, which is the port the relay targets. Keep `CLAUDE_MODEL` in `.env` equal to the one you start the helper with: it is part of the reply cache key.

Module 04 only: `docker compose up -d neo4j` (ready in ~30 s).

Stop with `docker compose down -v` and `kill %1`. Plain `down` leaves the anonymous Mongo and Neo4j volumes behind; `-v` frees them. Either way the data is gone; re-run ingestion and the graph build.

## Module 01

```sh
docker compose run --rm -T app python 01-naive-rag/ingestion.py   # ~5 min on CPU, 2022 chunks
docker compose run --rm -T app python 01-naive-rag/retrieval.py
docker compose run --rm app python 01-naive-rag/generation.py     # type a question
```

`EVAL_MAX_CASES` (default in `.env.example`: 1) caps the eval cases to limit Claude calls; leave it empty to run them all. Values of 0 or below are rejected.

Rebuild the image after changing `requirements.lock` or `Dockerfile`: `docker compose build app`. Re-lock with `./compile.sh`.

## Module 02

Ingest 02 before 03-05: it builds the metadata fields and the `metadata_filtered_index` those modules reuse.

```sh
docker compose run --rm -T app python 02-metadata-filtered/ingestion.py   # ~5 min on CPU; creates metadata_filtered_index with its filter fields
docker compose run --rm -T app python 02-metadata-filtered/retrieval.py
docker compose run --rm app python 02-metadata-filtered/generation.py     # pick 1-4 or type a question
docker compose run --rm -T app python 02-metadata-filtered/evals/precision_delta.py
docker compose run --rm -T app python 02-metadata-filtered/evals/latency.py
```

`EVAL_MAX_CASES` caps the eval cases here too.

## Module 03

Ingest 02 first: 03 builds BM25 in memory from the `metadata_filtered_rag` collection and queries `metadata_filtered_index`. It has no ingestion or evals of its own.

```sh
docker compose run --rm -T app python 03-hybrid-search/retrieval.py     # BM25, vector and fused RRF rankings
docker compose run --rm app python 03-hybrid-search/generation.py       # type a question
```

## Module 04

Needs the module 02 data and Neo4j (see Start). Neo4j has no host port; the app reaches it at `bolt://neo4j:7687`.

```sh
docker compose run --rm -T app python 04-graph-rag/graph_builder.py -n 20   # one Claude call per chunk
docker compose run --rm -T app python 04-graph-rag/retrieval.py
docker compose run --rm app python 04-graph-rag/generation.py               # type a question
```

Chunks load in ingestion order, so a small graph (`-n 20`) covers only the first letter; questions about other years fall back to vector results. Each retrieval query adds one Claude call to extract entities, so raise `LLM_MAX_CALLS` for a full build. Inspect the graph with `docker compose exec neo4j sh -c 'cypher-shell -u neo4j -p "${NEO4J_AUTH#neo4j/}"'`.

## Module 05

Needs only Mongo with module 02 ingested (it queries `metadata_filtered_index`); it does not use Neo4j.

```sh
docker compose run --rm app python 05-agentic-rag/agent.py        # type 1 (simple) or 3 (complex), or your own question
docker compose run --rm app python 05-agentic-rag/generation.py   # type a question
docker compose run --rm -T app python 05-agentic-rag/evals/tool_selection.py
docker compose run --rm -T app python 05-agentic-rag/evals/query_decomposition.py
docker compose run --rm -T app python 05-agentic-rag/evals/end_to_end.py
```

`MAX_ITERATIONS` (default in `.env.example`: 1; upstream default 3) caps the agent's retrieval attempts per sub-question, which bounds Claude calls. `EVAL_MAX_CASES` caps the eval cases as in module 01.

## Manual paste mode (fallback)

```sh
docker compose run --rm -e LLM_BACKEND=manual app python 01-naive-rag/generation.py
```

The prompt is saved to `.manual/NNN_prompt.txt` at the repo root and printed. Paste it into any chat, then paste the reply into the terminal followed by a line `END`. Replies are cached by backend, model, stop words and prompt. `LLM_MAX_CALLS` (default in `.env.example`: 50) caps successful uncached calls per process.

## What is substituted

| Tutorial | Sandbox |
|---|---|
| OpenAI embeddings (1536 dims) | Local `BAAI/bge-small-en-v1.5` (384 dims), baked into the image |
| MongoDB Atlas cloud + manual index step | `mongodb-atlas-local` container; the vector index is created by `ingestion.py` |
| Neo4j with a fixed default password and a host port | `neo4j` service in `compose.yaml`: random password from `init.sh`, internal network only |
| OpenAI chat model | Claude via `claude -p` through the helper (`LLM_BACKEND=claude-cli`), or manual paste |

Code edits are in `sandbox/shim/llm_shim.py` and the small `git diff main -- 01-naive-rag 02-metadata-filtered 03-hybrid-search 04-graph-rag 05-agentic-rag` (import fallback to the shim, plain Mongo URL without a TLS file, index creation, eval case cap via `EVAL_MAX_CASES`, `MAX_ITERATIONS` override in 05). The tutorial files still run unchanged outside the sandbox: `langchain_openai` is used when installed, and the Atlas TLS file and manual index instructions apply there.

## Security

- No runtime egress: the app container is on an internal network and reaches only Mongo and a fixed relay to the helper.
- Images pinned by digest; dependencies hash-locked (`--require-hashes`); the embedding model is pinned to a revision, safetensors only.
- App runs as a non-root user with all capabilities dropped, a read-only root filesystem, a `/tmp` tmpfs and a process limit. Module code is mounted read-only; only `.manual` is writable.
- The helper runs `claude -p` with no tools, no MCP servers and no settings or hooks, because prompts carry untrusted PDF text. The relay caps concurrent connections and enforces idle and overall deadlines.
- Mongo runs with no-new-privileges and a 3 GB memory limit. Its default capabilities are kept, because the Atlas Local image has not been verified to start without them.
- Neo4j runs as a non-root user with all capabilities dropped, no-new-privileges, a 1.5 GB memory limit and a random password; it is only on the internal network.
- Tracing is off.
- The helper listens on localhost only and requires a bearer token.
- The helper sets `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`, so `claude -p` loads no CLAUDE.md files (user, project or auto memory) into the prompt.

## Troubleshooting

- `401 Unauthorized` from the helper: the token in `.env` does not match the token the running helper loaded. `init.sh` warns about this. Fix with `LLM_HELPER_TOKEN=$(cat claude-helper/.token) ./init.sh`, restart the helper from `sandbox/claude-helper`, then recreate the app container.

## Usage notes

Generation uses your own Claude seat, so questions count against your usage. The helper is localhost-only: do not expose it, and do not share your `.env` or `claude-helper/.token`.
