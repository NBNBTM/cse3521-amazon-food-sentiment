#!/bin/zsh
# Author: Lindsey Yang
# Modified: 2026-11-03
# AI assistance: Codex assisted with this local launcher.
set -e
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'The local environment is missing. Follow Run locally in README.md.'
  read -r '?Press Enter to close.'
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.headless false
