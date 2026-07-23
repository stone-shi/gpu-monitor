# Agent Task: Local GPU & LM Studio Telemetry Monitor

See [CLAUDE.md](./CLAUDE.md) for repository commands, architecture, and conventions to follow while working on this task.

## Goal
Create a lightweight, highly stable background monitoring system that profiles local NVIDIA GPU usage alongside LM Studio token telemetry, committing all timeseries data into a PostgreSQL database backend for historical aggregation.

## System Components
1. **Database Schema Setup (`init_db.sql`)**: PostgreSQL tables tracking raw hardware performance snapshots and unique LLM token tracking data.
2. **Telemetry Background Daemon (`gpu_llm_monitor.py`)**: A dual-threaded Python service utilizing `pynvml` for hardware states and consuming `lms log stream` JSON records for token statistics.
3. **Service Management Configuration (`gpu_monitor.service`)**: A systemd service template ensuring the daemon automatically recovers, starts on boot, and runs cleanly in the background.

---

## 1. Database Schema (`init_db.sql`)
file .env contains the database information, you can just use it to create db and test.

Generate a script to initialize the following database structures. Ensure columns use proper indexing on `timestamp` fields for efficient date-truncation grouping (`day`, `week`, `month`).

```sql
-- Track hardware states sampled periodically
CREATE TABLE IF NOT EXISTS gpu_hardware_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    gpu_utilization INT NOT NULL,     -- 0 to 100 percentage
    vram_used_mb INT NOT NULL,        -- Expressed in MB
    temperature_c INT NOT NULL        -- Celcius
);

-- Track individual LLM inference actions
CREATE TABLE IF NOT EXISTS llm_token_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(255) NOT NULL,
    prompt_tokens INT NOT NULL,
    completion_tokens INT NOT NULL,
    total_tokens INT NOT NULL,
    tokens_per_sec FLOAT NOT NULL
);

-- Index optimization for time-series aggregations
CREATE INDEX IF NOT EXISTS idx_gpu_hardware_ts ON gpu_hardware_metrics (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_llm_token_ts ON llm_token_metrics (timestamp DESC);
```

## 2. Telemetry Ingestion Daemon (gpu_llm_monitor.py)
Implement a robust, production-ready daemon containing two operational worker threads feeding your local PostgreSQL node:

Core Requirements:
Thread 1 (Hardware Polling):

Initialize NVML using pynvml.nvmlInit().

Every 10 seconds, capture GPU Utilization percentage, VRAM utilization (convert bytes to Megabytes), and core temperature.

Commit these data points into gpu_hardware_metrics.

Thread 2 (LM Studio Log Processing):

Spawn a streaming subprocess executing: lms log stream --source model --filter output --json

Read lines from stdout sequentially. Parse incoming lines as JSON payloads.

Identify token execution reports by target fields (e.g., usage or stats). Extract model, input_tokens (or prompt), output_tokens (or completion), total_tokens, and tokens_per_sec.

Commit these stats safely to llm_token_metrics.

Technical Boundaries & Resiliency:
Use a centralized database pool or secure, independent connection instances per thread loop to avoid race conditions or broken pipes.

Wrap standard inputs/outputs and database queries in explicit try/except closures. If the database connection drops or lms restarts, the threads must log the error to stderr, sleep briefly, re-establish connection state, and continue execution without crashing the master daemon process.

## 3. Background Service Configuration (gpu_monitor.service)
Provide a standard Linux systemd service unit definition file to run this program continuously under user authority.
```
[Unit]
Description=NVIDIA GPU and LM Studio Token Telemetry Monitor Daemon
After=network.target postgresql.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /absolute/path/to/gpu_llm_monitor.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Definition of Done
[ ] init_db.sql successfully deploys tables and indexes.

[ ] gpu_llm_monitor.py starts without compilation errors and gracefully catches environment exceptions if NVML or PostgreSQL is temporarily unreachable.

[ ] The log stream consumer accurately extracts and isolates generation payloads from noisy CLI outputs.
