#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"

mkdir -p "$SCRIPT_DIR/test-reports"

echo "Running tests..."
pytest "$SCRIPT_DIR/tests" -v --junitxml="$SCRIPT_DIR/test-reports/results.xml"
