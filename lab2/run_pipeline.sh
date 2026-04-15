#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 scripts/fetch_data.py
python3 scripts/preprocess_data.py
python3 scripts/train_model.py
python3 scripts/evaluate_model.py
