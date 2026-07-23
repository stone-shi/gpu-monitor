import os
import streamlit as st
import pandas as pd
import psycopg2

# Set page config
st.set_page_config(
    page_title="GPU & LLM Telemetry Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 GPU & LocalAI Telemetry Dashboard")
st.markdown("Real-time monitoring of local hardware utilization and LLM inference telemetry.")

# Sidebar Controls for Auto-Refresh
st.sidebar.title("⚙️ Settings")
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh page", value=True, help="Toggle real-time database polling.")
refresh_interval = st.sidebar.slider("Polling Interval (seconds)", min_value=2, max_value=30, value=5)

# Load and display version info if available
version_hash = "unknown"
version_time = "unknown"
version_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.txt")
if os.path.exists(version_path):
    try:
        with open(version_path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    if k == "hash":
                        version_hash = v
                    elif k == "timestamp":
                        version_time = v
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Version**: `{version_hash}`")
st.sidebar.markdown(f"**Built**: `{version_time}`")

# Database connection helper
def get_db_connection():
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

# Cache data queries for performance (refresh every 5 seconds)
@st.cache_data(ttl=5)
def load_hardware_data():
    try:
        conn = get_db_connection()
        query = """
        SELECT timestamp, gpu_utilization, vram_used_mb, temperature_c 
        FROM gpu_hardware_metrics 
        ORDER BY timestamp DESC 
        LIMIT 300
        """
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading GPU hardware metrics: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_llm_data():
    try:
        conn = get_db_connection()
        query = """
        SELECT timestamp, model_name, prompt_tokens, completion_tokens, total_tokens, tokens_per_sec, prompt_text, response_text 
        FROM llm_token_metrics 
        ORDER BY timestamp DESC 
        LIMIT 100
        """
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading LLM token metrics: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_overall_gpu_active():
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            COUNT(*) as total_samples,
            SUM(CASE WHEN gpu_utilization > 50 THEN 1 ELSE 0 END) as active_samples
        FROM gpu_hardware_metrics
        """
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty and df.iloc[0]['total_samples'] > 0:
            total = df.iloc[0]['total_samples']
            active = df.iloc[0]['active_samples'] or 0
            return (active / total) * 100
        return 0.0
    except Exception as e:
        st.error(f"Error loading overall GPU active metric: {e}")
        return 0.0

@st.cache_data(ttl=5)
def load_daily_tokens():
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            date_trunc('day', timestamp) AS day,
            COUNT(*) AS requests,
            SUM(prompt_tokens) AS prompt_tokens,
            SUM(completion_tokens) AS completion_tokens,
            SUM(total_tokens) AS total_tokens
        FROM llm_token_metrics 
        GROUP BY day
        ORDER BY day ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty:
            df['day'] = pd.to_datetime(df['day']).dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        st.error(f"Error loading daily token data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_monthly_tokens():
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            date_trunc('month', timestamp) AS month,
            COUNT(*) AS requests,
            SUM(prompt_tokens) AS prompt_tokens,
            SUM(completion_tokens) AS completion_tokens,
            SUM(total_tokens) AS total_tokens
        FROM llm_token_metrics 
        GROUP BY month
        ORDER BY month ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty:
            df['month'] = pd.to_datetime(df['month']).dt.strftime('%Y-%m')
        return df
    except Exception as e:
        st.error(f"Error loading monthly token data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_request_count():
    try:
        conn = get_db_connection()
        query = "SELECT COUNT(*) as total_requests FROM llm_token_metrics"
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty:
            return int(df.iloc[0]['total_requests'])
        return 0
    except Exception as e:
        st.error(f"Error loading request count: {e}")
        return 0

# Load datasets
df_hw = load_hardware_data()
df_llm = load_llm_data()
df_daily = load_daily_tokens()
df_monthly = load_monthly_tokens()
overall_gpu_active = load_overall_gpu_active()
total_requests = load_request_count()

# Metric summary row
col1, col2, col3, col4, col5, col6 = st.columns(6)

if not df_hw.empty:
    latest_hw = df_hw.iloc[0]
    col1.metric("GPU Utilization", f"{latest_hw['gpu_utilization']}%")
    col2.metric("VRAM Used", f"{latest_hw['vram_used_mb']:,} MB")
    col3.metric("GPU Temperature", f"{latest_hw['temperature_c']}°C")
else:
    col1.metric("GPU Utilization", "N/A")
    col2.metric("VRAM Used", "N/A")
    col3.metric("GPU Temperature", "N/A")

col4.metric("Overall GPU Active (>50%)", f"{overall_gpu_active:.1f}%")
col5.metric("Total Requests", f"{total_requests:,}")

if not df_llm.empty:
    avg_speed = df_llm['tokens_per_sec'].mean()
    col6.metric("Avg Generation Speed", f"{avg_speed:.2f} t/s")
else:
    col6.metric("Avg Generation Speed", "0.00 t/s")

st.markdown("---")

# Layout Tabs
tab1, tab2, tab3 = st.tabs(["🖥️ GPU Hardware Performance", "🤖 Live LLM Inferences", "📈 Historical Token Usage"])

with tab1:
    st.subheader("GPU Metrics Over Time")
    if not df_hw.empty:
        # Sort ascending for chronological graph plots
        df_hw_sorted = df_hw.sort_values("timestamp")
        
        # Dual columns for graphs
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("**GPU Core Utilization (%)**")
            st.line_chart(df_hw_sorted.set_index("timestamp")[["gpu_utilization"]], height=250)
        with g_col2:
            st.markdown("**VRAM Usage (MB)**")
            st.line_chart(df_hw_sorted.set_index("timestamp")[["vram_used_mb"]], height=250)
            
        st.markdown("**Core Temperature (°C)**")
        st.line_chart(df_hw_sorted.set_index("timestamp")[["temperature_c"]], height=200)
    else:
        st.info("No GPU metrics captured yet. Ensure the python daemon is running.")

with tab2:
    st.subheader("Recent Generation Transcripts")
    if not df_llm.empty:
        # Format timestamps for display
        df_llm_disp = df_llm.copy()
        df_llm_disp['timestamp'] = df_llm_disp['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format numeric columns with commas for readability
        df_llm_fmt = df_llm_disp[["timestamp", "model_name", "prompt_tokens", "completion_tokens", "total_tokens", "tokens_per_sec"]].copy()
        for col in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            df_llm_fmt[col] = df_llm_fmt[col].apply(lambda x: f"{x:,}")
        df_llm_fmt["tokens_per_sec"] = df_llm_fmt["tokens_per_sec"].apply(lambda x: f"{x:,.2f}")
        
        st.dataframe(
            df_llm_fmt,
            use_container_width=True
        )
        
        st.markdown("### 🔍 Generation Transcript Inspector")
        # Selector for generation log
        selected_index = st.selectbox(
            "Select a request to inspect:",
            df_llm_disp.index,
            format_func=lambda idx: f"[{df_llm_disp.loc[idx, 'timestamp']}] {df_llm_disp.loc[idx, 'model_name']} ({df_llm_disp.loc[idx, 'total_tokens']:,} tokens)"
        )
        
        selected_run = df_llm_disp.loc[selected_index]
        det_col1, det_col2 = st.columns(2)
        with det_col1:
            st.markdown("**Prompt Text**")
            st.text_area("prompt_viewer", selected_run['prompt_text'], height=250, label_visibility="collapsed")
        with det_col2:
            st.markdown("**Response Text**")
            st.text_area("response_viewer", selected_run['response_text'], height=250, label_visibility="collapsed")
    else:
        st.info("No LLM inferences logged yet. Generate text using LocalAI to populate.")

def format_human(n):
    """Format a number into human-readable K, M, B notation."""
    if n is None:
        return "0"
    n = float(n)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return f"{int(n)}"

with tab3:
    st.subheader("Historical Token Usage Patterns")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("**Daily Token Usage**")
        if not df_daily.empty:
            # Display daily tokens as stacked bar chart
            st.bar_chart(
                df_daily.set_index("day")[["prompt_tokens", "completion_tokens"]],
                use_container_width=True
            )
            # Format table numbers to K/M/B notation
            df_daily_fmt = df_daily.sort_values("day", ascending=False)[["day", "requests", "prompt_tokens", "completion_tokens", "total_tokens"]].copy()
            df_daily_fmt["requests"] = df_daily_fmt["requests"].apply(lambda x: f"{int(x):,}")
            for col in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                df_daily_fmt[col] = df_daily_fmt[col].apply(format_human)
            st.dataframe(
                df_daily_fmt,
                use_container_width=True
            )
        else:
            st.info("No daily token usage data found.")
            
    with col_t2:
        st.markdown("**Monthly Token Usage**")
        if not df_monthly.empty:
            # Display monthly tokens as stacked bar chart
            st.bar_chart(
                df_monthly.set_index("month")[["prompt_tokens", "completion_tokens"]],
                use_container_width=True
            )
            # Format table numbers to K/M/B notation
            df_monthly_fmt = df_monthly.sort_values("month", ascending=False)[["month", "requests", "prompt_tokens", "completion_tokens", "total_tokens"]].copy()
            df_monthly_fmt["requests"] = df_monthly_fmt["requests"].apply(lambda x: f"{int(x):,}")
            for col in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                df_monthly_fmt[col] = df_monthly_fmt[col].apply(format_human)
            st.dataframe(
                df_monthly_fmt,
                use_container_width=True
            )
        else:
            st.info("No monthly token usage data found.")

# Real-time auto-refresh executor (placed at the end to sleep after rendering)
if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()
