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
    tokens_per_sec FLOAT NOT NULL,
    prompt_text TEXT,
    response_text TEXT
);

-- Index optimization for time-series aggregations
CREATE INDEX IF NOT EXISTS idx_gpu_hardware_ts ON gpu_hardware_metrics (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_llm_token_ts ON llm_token_metrics (timestamp DESC);
