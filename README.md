# Local GPU & LocalAI Telemetry Monitor

A lightweight, stable background monitoring system that profiles local NVIDIA GPU usage alongside LocalAI token telemetry, committing all timeseries data into a PostgreSQL database for historical analysis and aggregation.

---

## Features

- **GPU Hardware Polling (Thread 1)**: Utilizes NVIDIA Management Library (`pynvml`) to capture GPU utilization %, VRAM usage (MB), and core temperature every 10 seconds.
- **LocalAI Telemetry Ingestion (Thread 2)**: Polls LocalAI's `/api/traces` HTTP endpoint, decodes the request/response bodies of each chat completion, parses token usage and generation speed (including streamed responses), and writes complete records (with prompts/responses) to PostgreSQL.
- **Fail-Safe & Auto-Recovery**: Robust exception isolation per thread loop handles network fluctuations, PostgreSQL connection drops, and LocalAI restarts without crashing the master daemon.
- **Systemd Daemon Integration**: Configured to run under user authority with automatic boot-time startup and recovery policies.
- **Formatted CLI Inspector**: Includes a custom CLI utility (`query_stats.py`) to query and aggregate historical data.

---

## File Structure

- [gpu_llm_monitor.py](file:///home/stoneshi/data/src/stone/gpu-monitor/gpu_llm_monitor.py): The primary daemon running the concurrent telemetry polling loops.
- [init_db.sql](file:///home/stoneshi/data/src/stone/gpu-monitor/init_db.sql): SQL script initializing the tables and descending timeseries indexes.
- [gpu_monitor.service](file:///home/stoneshi/data/src/stone/gpu-monitor/gpu_monitor.service): systemd service definition template.
- [query_stats.py](file:///home/stoneshi/data/src/stone/gpu-monitor/query_stats.py): Command-line utility to query recent telemetry and aggregated stats.
- [.env](file:///home/stoneshi/data/src/stone/gpu-monitor/.env): Local environment file storing database credentials (git-ignored).

---

## Requirements

The daemon requires the following dependencies:
- **Operating System**: Linux (Ubuntu/Debian recommended)
- **NVIDIA GPU** with drivers installed
- **LocalAI** running and reachable over HTTP (its `/api/traces` endpoint)
- **PostgreSQL** server running locally
- **Python Modules**:
  ```bash
  sudo apt-get install -y python3-pynvml python3-psycopg2
  ```

---

## Installation & Setup

### 1. Database Initialization
Ensure PostgreSQL is running. Configure credentials in your local `.env` file, then initialize the schema:
```bash
# Connect to PostgreSQL and execute:
python3 -c "
import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='YOUR_PASSWORD', database='gpu-monitor')
cur = conn.cursor()
with open('init_db.sql', 'r') as f:
    cur.execute(f.read())
conn.commit(); cur.close(); conn.close()
"
```

### 2. Installing the Background Daemon
Install the systemd configuration file to run the daemon continuously:

```bash
# 1. Copy the service unit config to systemd directory
sudo cp gpu_monitor.service /etc/systemd/system/

# 2. Reload systemd manager configurations
sudo systemctl daemon-reload

# 3. Enable the service to run on system boot and start it
sudo systemctl enable --now gpu_monitor.service
```

---

## Monitoring and Management

### Service Logs and Status
Check the status of the background daemon:
```bash
systemctl status gpu_monitor.service
```

View the live streaming logs of the daemon using `journalctl`:
```bash
sudo journalctl -u gpu_monitor.service -f
```

### Querying Telemetry Data
A query command-line interface helper is included. Run it to inspect recent hardware states, LLM inferences, and aggregated model performances:
```bash
./query_stats.py
```

### Database Schema
The database uses two primary tables optimised with descending indexes on `timestamp`:

#### `gpu_hardware_metrics`
- `id` (SERIAL PRIMARY KEY)
- `timestamp` (TIMESTAMPTZ)
- `gpu_utilization` (INT, 0-100%)
- `vram_used_mb` (INT, MB)
- `temperature_c` (INT, °C)

#### `llm_token_metrics`
- `id` (SERIAL PRIMARY KEY)
- `timestamp` (TIMESTAMPTZ)
- `model_name` (VARCHAR)
- `prompt_tokens` (INT)
- `completion_tokens` (INT)
- `total_tokens` (INT)
- `tokens_per_sec` (FLOAT)
- `prompt_text` (TEXT, Nullable)
- `response_text` (TEXT, Nullable)
