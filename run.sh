#!/usr/bin/env bash
# Convenience launcher for the Streamlit dashboard.
# Uses uv to ensure the correct virtualenv without manual activation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Ukraine Air Raid Analysis Dashboard..."
echo "Open http://localhost:8501 in your browser."
echo ""

uv run --project "$SCRIPT_DIR" streamlit run "$SCRIPT_DIR/app/Home.py" "$@"
