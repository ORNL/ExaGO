#!/bin/bash
# AgentiGrid Launcher — start script
# Ensures cwd is the project root so config paths resolve correctly.
# Uses 'python -m streamlit' to ensure the active conda/venv Python is used.

# Uncomment and adjust if using a conda environment:
# source activate exago312

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1
echo "Starting AgentiGrid Launcher from: $PROJECT_ROOT"
python -m streamlit run launcher/app.py "$@"
