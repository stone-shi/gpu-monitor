#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
import threading
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

def lms_log_processing_thread():
    """Thread 2: Consumes LM Studio log stream and commits LLM token and text metrics to database."""
    print("Starting LM Studio Log Processing Thread...", file=sys.stderr)
    conn = None
    last_prompt = None
    
    while True:
        proc = None
        try:
            print("Spawning lms log stream subprocess...", file=sys.stderr)
            proc = subprocess.Popen(
                ["lms", "log", "stream", "--source", "model", "--filter", "input,output", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Read streaming output line by line
            while True:
                line = proc.stdout.readline()
                if not line:
                    print("lms log stream subprocess output ended.", file=sys.stderr)
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Try parsing line as JSON payload
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Check target fields for token execution reports
                data_obj = payload.get("data", {})
                if not isinstance(data_obj, dict) or not data_obj:
                    data_obj = payload
                
                event_type = data_obj.get("type")
                
                # If it's an input event, save the prompt text and continue
                if event_type == "llm.prediction.input":
                    last_prompt = data_obj.get("input")
                    continue
                
                # If it's an output event, process metrics and insert
                if event_type == "llm.prediction.output" or "stats" in data_obj or "usage" in data_obj:
                    stats = data_obj.get("stats", {}) or data_obj.get("usage", {})
                    if not isinstance(stats, dict):
                        stats = {}
                    
                    model_name = data_obj.get("modelIdentifier") or data_obj.get("model") or payload.get("model")
                    prompt_tokens = stats.get("promptTokensCount") or stats.get("prompt_tokens") or stats.get("input_tokens") or stats.get("prompt")
                    completion_tokens = stats.get("predictedTokensCount") or stats.get("completion_tokens") or stats.get("output_tokens") or stats.get("completion")
                    total_tokens = stats.get("totalTokensCount") or stats.get("total_tokens")
                    tokens_per_sec = stats.get("tokensPerSecond") or stats.get("tokens_per_sec") or stats.get("t/s")
                    
                    response_text = data_obj.get("output")
                    
                    # Cast metrics safely
                    try:
                        if prompt_tokens is not None:
                            prompt_tokens = int(prompt_tokens)
                    except (TypeError, ValueError):
                        prompt_tokens = None
                        
                    try:
                        if completion_tokens is not None:
                            completion_tokens = int(completion_tokens)
                    except (TypeError, ValueError):
                        completion_tokens = None
                        
                    try:
                        if total_tokens is not None:
                            total_tokens = int(total_tokens)
                    except (TypeError, ValueError):
                        total_tokens = None
                    
                    try:
                        if tokens_per_sec is not None:
                            tokens_per_sec = float(tokens_per_sec)
                    except (TypeError, ValueError):
                        tokens_per_sec = 0.0

                    # If we have minimum valid fields, insert into database
                    if model_name and prompt_tokens is not None and completion_tokens is not None:
                        if total_tokens is None:
                            total_tokens = prompt_tokens + completion_tokens
                        if tokens_per_sec is None:
                            tokens_per_sec = 0.0
                            
                        db_inserted = False
                        while not db_inserted:
                            try:
                                if conn is None or conn.closed:
                                    conn = get_db_connection()
                                    print("LM Studio Thread connected to database.", file=sys.stderr)
                                
                                with conn.cursor() as cur:
                                    cur.execute(
                                        """
                                        INSERT INTO llm_token_metrics 
                                        (model_name, prompt_tokens, completion_tokens, total_tokens, tokens_per_sec, prompt_text, response_text)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                                        """,
                                        (model_name, prompt_tokens, completion_tokens, total_tokens, tokens_per_sec, last_prompt, response_text)
                                    )
                                conn.commit()
                                db_inserted = True
                                # Clear prompt once successfully logged
                                last_prompt = None
                            except Exception as e:
                                print(f"LM Studio Thread DB insertion failed: {e}. Closing connection.", file=sys.stderr)
                                try:
                                    if conn:
                                        conn.close()
                                except Exception:
                                    pass
                                conn = None
                                # Sleep briefly before retrying database insertion
                                time.sleep(2)
                
        except Exception as e:
            print(f"Error in LM Studio thread loop: {e}", file=sys.stderr)
        
        # Clean up subprocess if alive
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        
        print("LM Studio subprocess connection lost or failed. Re-trying in 5 seconds...", file=sys.stderr)
        time.sleep(5)

def main():
    # Load environment variables from the .env next to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    load_env(env_path)
    
    print("Starting Telemetry Ingestion Daemon...", file=sys.stderr)
    
    # Run the worker threads
    t1 = threading.Thread(target=hardware_polling_thread, name="HardwarePolling", daemon=True)
    t2 = threading.Thread(target=lms_log_processing_thread, name="LMSLogProcessing", daemon=True)
    
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
