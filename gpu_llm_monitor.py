#!/usr/bin/env python3
import os
import sys
import time
import json
import base64
import threading
import collections
import urllib.request
import urllib.error
from datetime import datetime
import psycopg2
import pynvml

def load_env(env_path):
    """Loads environment variables from a .env file."""
    if not os.path.exists(env_path):
        print(f"Warning: .env file not found at {env_path}", file=sys.stderr)
        return
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                os.environ[key] = val

def get_db_connection():
    """Establishes and returns a database connection using environment variables."""
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "gpu-monitor")
    
    return psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=dbname
    )

def hardware_polling_thread():
    """Thread 1: Polls GPU utilization, VRAM usage, and core temperature every 10 seconds."""
    print("Starting Hardware Polling Thread...", file=sys.stderr)
    conn = None
    
    # Initialize NVML
    nvml_initialized = False
    while not nvml_initialized:
        try:
            pynvml.nvmlInit()
            nvml_initialized = True
            print("NVML initialized successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing NVML: {e}. Retrying in 5 seconds...", file=sys.stderr)
            time.sleep(5)

    try:
        # Fetching handle for the first GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"Error getting GPU handle: {e}. Hardware Polling Thread exiting.", file=sys.stderr)
        return

    while True:
        try:
            # Check or establish database connection
            if conn is None or conn.closed:
                try:
                    conn = get_db_connection()
                    print("Hardware Polling Thread connected to database.", file=sys.stderr)
                except Exception as e:
                    print(f"Hardware Polling DB connection failed: {e}. Will retry on next sample.", file=sys.stderr)
                    time.sleep(10)
                    continue

            # Gather GPU metrics using pynvml
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
                
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used_mb = int(mem.used / (1024 * 1024))
                
                temperature_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception as e:
                print(f"NVML metrics gathering failed: {e}", file=sys.stderr)
                time.sleep(10)
                continue

            # Commit the hardware metrics
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO gpu_hardware_metrics (gpu_utilization, vram_used_mb, temperature_c)
                        VALUES (%s, %s, %s)
                        """,
                        (gpu_utilization, vram_used_mb, temperature_c)
                    )
                conn.commit()
            except Exception as e:
                print(f"Failed to commit hardware metrics to DB: {e}. Closing connection.", file=sys.stderr)
                try:
                    if conn:
                        conn.close()
                except Exception:
                    pass
                conn = None

        except Exception as e:
            print(f"Unexpected error in hardware polling thread: {e}", file=sys.stderr)
            
        time.sleep(10)

def fetch_traces(url, api_key, timeout=10):
    """Fetches the current LocalAI /api/traces buffer (newest-first list of request/response records)."""
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def parse_trace_timestamp(ts_raw):
    """Parses LocalAI's nanosecond-precision RFC3339 timestamp, truncated to microsecond precision."""
    if not ts_raw:
        return None
    ts_raw = ts_raw.strip()
    if ts_raw.endswith("Z"):
        ts_raw = ts_raw[:-1] + "+00:00"
    if "." in ts_raw:
        base, frac_and_tz = ts_raw.split(".", 1)
        tz_idx = next((i for i, ch in enumerate(frac_and_tz) if ch in "+-"), len(frac_and_tz))
        frac, tz = frac_and_tz[:tz_idx], frac_and_tz[tz_idx:]
        ts_raw = f"{base}.{frac[:6]}{tz}"
    try:
        return datetime.fromisoformat(ts_raw)
    except ValueError:
        return None

def extract_last_user_message(messages):
    """Extracts the most recent user message's text from an OpenAI-style chat `messages` array."""
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [part.get("text") for part in content if isinstance(part, dict) and part.get("type") == "text"]
            return "\n".join(t for t in texts if t) or None
    return None

def parse_streaming_response(body_text):
    """Reconstructs (trace_id, model_name, response_text, usage) from an SSE chat.completion.chunk stream."""
    trace_id = None
    model_name = None
    usage = None
    content_parts = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        trace_id = trace_id or chunk.get("id")
        model_name = model_name or chunk.get("model")
        usage = chunk.get("usage") or usage
        choices = chunk.get("choices") or []
        if choices:
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                content_parts.append(piece)
    return trace_id, model_name, "".join(content_parts), usage

def parse_json_response(body_text):
    """Extracts (trace_id, model_name, response_text, usage) from a non-streaming chat.completion body."""
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError:
        return None, None, None, None
    choices = body.get("choices") or []
    response_text = (choices[0].get("message") or {}).get("content") if choices else None
    return body.get("id"), body.get("model"), response_text, body.get("usage")

def process_trace(entry):
    """Turns one /api/traces entry into an llm_token_metrics record, or None if it isn't a usable chat completion."""
    request = entry.get("request") or {}
    response = entry.get("response") or {}
    if request.get("path") != "/v1/chat/completions" or response.get("status") != 200:
        return None

    try:
        req_body = json.loads(base64.b64decode(request.get("body", "")).decode("utf-8", "replace"))
    except (ValueError, json.JSONDecodeError):
        req_body = {}

    resp_raw = base64.b64decode(response.get("body", "")).decode("utf-8", "replace")
    if resp_raw.lstrip().startswith("data:"):
        trace_id, model_name, response_text, usage = parse_streaming_response(resp_raw)
    else:
        trace_id, model_name, response_text, usage = parse_json_response(resp_raw)

    if not trace_id:
        return None

    model_name = model_name or req_body.get("model") or "unknown"
    prompt_text = extract_last_user_message(req_body.get("messages"))

    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is None:
        prompt_tokens = len(prompt_text) // 4 if prompt_text else 0
    if completion_tokens is None:
        completion_tokens = len(response_text) // 4 if response_text else 0
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    duration_s = (entry.get("duration") or 0) / 1_000_000_000
    tokens_per_sec = (completion_tokens / duration_s) if duration_s > 0 and completion_tokens else 0.0

    return {
        "trace_id": trace_id,
        "timestamp": parse_trace_timestamp(entry.get("timestamp")),
        "model_name": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "prompt_text": prompt_text,
        "response_text": response_text,
    }

def localai_trace_polling_thread():
    """Thread 2: Polls LocalAI's /api/traces endpoint and commits LLM token and text metrics to database."""
    print("Starting LocalAI Trace Polling Thread...", file=sys.stderr)
    traces_url = os.environ.get("LOCALAI_TRACES_URL", "")
    api_key = os.environ.get("LOCALAI_API_KEY", "")
    poll_interval = float(os.environ.get("LOCALAI_POLL_INTERVAL", "5"))

    if not traces_url:
        print("LOCALAI_TRACES_URL not set. LocalAI Trace Polling Thread exiting.", file=sys.stderr)
        return

    conn = None
    # Bounded FIFO of trace ids already handled, so a fresh /api/traces poll never
    # double-inserts an entry still sitting in LocalAI's ring buffer.
    seen_ids = set()
    seen_order = collections.deque()
    MAX_SEEN = 5000

    def mark_seen(trace_id):
        seen_ids.add(trace_id)
        seen_order.append(trace_id)
        if len(seen_order) > MAX_SEEN:
            seen_ids.discard(seen_order.popleft())

    # Bootstrap: mark whatever is already in the trace buffer as seen without inserting it,
    # so restarting the daemon doesn't replay completions already recorded on a prior run.
    try:
        for entry in fetch_traces(traces_url, api_key):
            record = process_trace(entry)
            if record:
                mark_seen(record["trace_id"])
    except Exception as e:
        print(f"Initial LocalAI trace fetch failed: {e}", file=sys.stderr)

    while True:
        try:
            traces = fetch_traces(traces_url, api_key)
        except Exception as e:
            print(f"Failed to fetch LocalAI traces: {e}. Retrying in {poll_interval}s...", file=sys.stderr)
            time.sleep(poll_interval)
            continue

        # Traces come back newest-first; walk oldest-first so inserts land in chronological order.
        for entry in reversed(traces):
            try:
                record = process_trace(entry)
            except Exception as e:
                print(f"Failed to parse LocalAI trace entry: {e}", file=sys.stderr)
                continue

            if not record or record["trace_id"] in seen_ids:
                continue
            mark_seen(record["trace_id"])

            db_inserted = False
            while not db_inserted:
                try:
                    if conn is None or conn.closed:
                        conn = get_db_connection()
                        print("LocalAI Trace Thread connected to database.", file=sys.stderr)

                    with conn.cursor() as cur:
                        if record["timestamp"] is not None:
                            cur.execute(
                                """
                                INSERT INTO llm_token_metrics
                                (timestamp, model_name, prompt_tokens, completion_tokens, total_tokens, tokens_per_sec, prompt_text, response_text)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (record["timestamp"], record["model_name"], record["prompt_tokens"], record["completion_tokens"],
                                 record["total_tokens"], record["tokens_per_sec"], record["prompt_text"], record["response_text"])
                            )
                        else:
                            cur.execute(
                                """
                                INSERT INTO llm_token_metrics
                                (model_name, prompt_tokens, completion_tokens, total_tokens, tokens_per_sec, prompt_text, response_text)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """,
                                (record["model_name"], record["prompt_tokens"], record["completion_tokens"],
                                 record["total_tokens"], record["tokens_per_sec"], record["prompt_text"], record["response_text"])
                            )
                    conn.commit()
                    db_inserted = True
                except Exception as e:
                    print(f"LocalAI Trace Thread DB insertion failed: {e}. Closing connection.", file=sys.stderr)
                    try:
                        if conn:
                            conn.close()
                    except Exception:
                        pass
                    conn = None
                    time.sleep(2)

        time.sleep(poll_interval)

def main():
    # Load environment variables from the .env next to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    load_env(env_path)
    
    print("Starting Telemetry Ingestion Daemon...", file=sys.stderr)
    
    # Run the worker threads
    t1 = threading.Thread(target=hardware_polling_thread, name="HardwarePolling", daemon=True)
    t2 = threading.Thread(target=localai_trace_polling_thread, name="LocalAITracePolling", daemon=True)
    
    t1.start()
    t2.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down daemon...", file=sys.stderr)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
