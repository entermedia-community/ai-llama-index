#!/bin/bash

# Llama index launcher script

# All configurations are loaded from .env file in the same directory as this script or in /root/.env

# Load environment variables from server root if .env exists
if [ -f "/root/.env" ]; then
    set -a
    source "/root/.env"
    set +a
elif [ -f "$(dirname "$0")/.env" ]; then
    set -a
    source "$(dirname "$0")/.env"
    set +a
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
#export CUDA_VISIBLE_DEVICES=0

# Kill any existing processes and free up port 4600
lsof -ti :4600 | xargs -r kill -9

sleep 5

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

LOGFILE=/root/logs/llamaindex
mkdir -p "$LOGFILE"

python -m uvicorn main:app \
	--host 0.0.0.0 \
	--port 4600 \
	--timeout-keep-alive 120 \
	--workers 1  \
	2>&1 | multilog t s5000000 n3 "$LOGFILE" &

deactivate