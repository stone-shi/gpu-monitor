FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl for health check, libpq for postgres interactions)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    psycopg2-binary

# Copy the dashboard code
COPY dashboard.py /app/dashboard.py
COPY version.txt* /app/

# Expose Streamlit's default port
EXPOSE 8501

# Configure health check for Docker
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
