#!/usr/bin/env python3
import os
import sys
import psycopg2

def load_env(env_path):
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip().strip("'\"")

def get_db_connection():
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "gpu-monitor")
    return psycopg2.connect(host=host, port=port, user=user, password=password, database=dbname)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_env(os.path.join(script_dir, '.env'))
    
    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()
    
    print("=" * 70)
    print(" RECENT GPU HARDWARE METRICS (Last 5 Samples)")
    print("=" * 70)
    cur.execute("SELECT timestamp, gpu_utilization, vram_used_mb, temperature_c FROM gpu_hardware_metrics ORDER BY timestamp DESC LIMIT 5")
    rows = cur.fetchall()
    if not rows:
        print("No GPU metrics recorded yet.")
    for r in rows:
        print(f"{r[0].strftime('%Y-%m-%d %H:%M:%S')} | GPU: {r[1]:>3}% | VRAM: {r[2]:>5} MB | Temp: {r[3]:>2}°C")
        
    print("\n" + "=" * 70)
    print(" RECENT LLM INFERENCE METRICS (Last 5 Generations)")
    print("=" * 70)
    cur.execute("SELECT timestamp, model_name, prompt_tokens, completion_tokens, total_tokens, tokens_per_sec, prompt_text, response_text FROM llm_token_metrics ORDER BY timestamp DESC LIMIT 5")
    rows = cur.fetchall()
    if not rows:
        print("No LLM token metrics recorded yet.")
    for r in rows:
        print(f"{r[0].strftime('%Y-%m-%d %H:%M:%S')} | {r[1][:25]:<25} | Prompt: {r[2]:>4} tok | Compl: {r[3]:>4} tok | Total: {r[4]:>4} tok | Speed: {r[5]:>6.2f} tok/s")
        prompt_snippet = (r[6] or "").strip().replace("\n", " ")
        if len(prompt_snippet) > 60:
            prompt_snippet = prompt_snippet[:57] + "..."
        response_snippet = (r[7] or "").strip().replace("\n", " ")
        if len(response_snippet) > 60:
            response_snippet = response_snippet[:57] + "..."
        print(f"  └─ Prompt:   {prompt_snippet}")
        print(f"  └─ Response: {response_snippet}")
        print("-" * 70)
        
    print("\n" + "=" * 70)
    print(" AGGREGATED METRICS BY MODEL")
    print("=" * 70)
    cur.execute("""
        SELECT 
            model_name, 
            COUNT(*) as total_calls, 
            SUM(total_tokens) as total_tokens, 
            ROUND(AVG(tokens_per_sec)::numeric, 2) as avg_speed 
        FROM llm_token_metrics 
        GROUP BY model_name
    """)
    rows = cur.fetchall()
    if not rows:
        print("No LLM aggregations available.")
    for r in rows:
        print(f"Model: {r[0]:<30} | Requests: {r[1]:>3} | Total Tokens: {r[2]:>6} | Avg Speed: {r[3]:>6.2f} tok/s")
    print("=" * 70)

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
