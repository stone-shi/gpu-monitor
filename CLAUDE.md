# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A lightweight daemon that profiles local NVIDIA GPU usage and LocalAI LLM token telemetry, writing timeseries data to PostgreSQL, plus a Streamlit dashboard and CLI tool for viewing it.

## Commands

```bash
# Run tests (creates .venv, installs requirements.txt, runs pytest with junit output)
./test.sh

# Run a single test file / test
.venv/bin/pytest tests/test_extract_truncated_json.py -v
.venv/bin/pytest tests/test_threads.py::TestHardwarePollingThread::test_nvml_init_failure_retries -v

# Build and push the dashboard Docker image (tags with local git hash, pushes to registry.shifamily.com)
./build.sh

# Run the telemetry daemon directly (expects a .env file next to the script; requires pynvml + a real GPU)
python3 gpu_llm_monitor.py

# Query recent stats from the CLI
./query_stats.py

# Run the dashboard locally
streamlit run dashboard.py
```

There is no separate lint config; keep new Python consistent with the existing style (stdlib logging via `print(..., file=sys.stderr)`, no type hints, no formatter configured).

## Architecture

Three independently-runnable Python entry points share the same DB access pattern and no common module — `get_db_connection()` and `.env`-loading (`load_env()`) are duplicated in `gpu_llm_monitor.py`, `dashboard.py`, and `query_stats.py`. If you change the connection/env logic, update all three.

- **`gpu_llm_monitor.py`** — the daemon (`main()`), meant to run under systemd (see `gpu_monitor.service`). Loads `.env` from its own directory, then starts two daemon threads that run forever and never share state except through Postgres:
  - `hardware_polling_thread()`: initializes NVML, polls GPU 0 every 10s (utilization/VRAM/temp), inserts into `gpu_hardware_metrics`. Retries NVML init in a loop; reconnects to Postgres if the connection drops or a commit fails.
  - `localai_trace_polling_thread()`: polls LocalAI's `/api/traces` HTTP endpoint (a ring buffer of recent request/response records that LocalAI keeps in-process) instead of tailing a log stream. LocalAI runs as its own Docker container (formerly this was LM Studio via `lms log stream`; that integration is gone). This is the trickiest part of the codebase:
    - Each trace entry has base64-encoded `request.body`/`response.body`. `process_trace()` decodes and JSON-parses both, keeping only `POST /v1/chat/completions` entries with a `200` status.
    - Non-streaming responses are a single `chat.completion` JSON object (`parse_json_response()`); streaming responses are an SSE `chat.completion.chunk` stream (`parse_streaming_response()`) that must be reassembled by concatenating each chunk's `delta.content` and pulling `usage` off whichever chunk carries it (only present if the client sent `stream_options.include_usage: true`). Both paths fall back to `len(text) // 4` token estimates when `usage` is absent.
    - `extract_last_user_message()` pulls the prompt text out of the request's OpenAI-style `messages` array (last `role: "user"` entry), since LocalAI's trace format has no single flat prompt string like LM Studio did.
    - Because `/api/traces` is a bounded ring buffer polled repeatedly rather than a stream that's consumed once, a bounded FIFO of already-inserted trace ids (`seen_ids`/`seen_order`, keyed by the completion's `id` field) prevents the same entry from being inserted twice across polls. On startup the thread does one throwaway fetch to mark whatever's already in the buffer as seen, so a daemon restart doesn't replay completions from a prior run.

- **`dashboard.py`** — Streamlit app with `@st.cache_data(ttl=5)`-decorated query functions, each opening/closing its own `psycopg2` connection. Self-driven auto-refresh: it calls `time.sleep(refresh_interval)` then `st.rerun()` at the end of the script rather than using a Streamlit-native refresh mechanism. Reads `version.txt` (produced by `build.sh`) to display build hash/timestamp in the sidebar.

- **`query_stats.py`** — standalone CLI that prints recent hardware samples, recent LLM generations (with truncated prompt/response snippets), and per-model aggregates. No daemon interaction; reads directly from Postgres.

- **`init_db.sql`** — creates `gpu_hardware_metrics` and `llm_token_metrics` (the latter has nullable `prompt_text`/`response_text` for full transcripts) plus descending timestamp indexes for `date_trunc`-based aggregation.

## Configuration

All three Python entry points read DB config from environment variables (with defaults), loaded from a `.env` file in the script's own directory: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`. `.env` is git-ignored and not checked in.

`gpu_llm_monitor.py` additionally reads `LOCALAI_TRACES_URL` (e.g. `http://localhost:4012/api/traces`; required — `localai_trace_polling_thread()` exits immediately if unset), `LOCALAI_API_KEY` (optional bearer token, only needed if LocalAI has auth enabled), and `LOCALAI_POLL_INTERVAL` (seconds between polls, default `5`).

Docker deployment (`docker-compose.yml`) only packages the dashboard, not the daemon — the daemon is meant to run on bare metal (via `gpu_monitor.service`) with direct GPU/NVML access, reaching LocalAI's `/api/traces` over the network, while the dashboard connects out to Postgres (by default via `host.docker.internal`, i.e. connecting to a Postgres instance on the host, not the container network).

## Tests

`tests/` uses pytest with mocks (`unittest.mock.patch`) against `pynvml`, `psycopg2`, and `fetch_traces` — no real GPU, DB, or LocalAI instance is needed to run the suite. Thread-target functions (`hardware_polling_thread`, `localai_trace_polling_thread`) run infinite loops, so tests break out of them by making a mocked `time.sleep` raise `KeyboardInterrupt` (or similar) after N calls — see `tests/test_threads.py` for the pattern to follow when adding new thread-loop tests.
