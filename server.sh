#!/bin/bash

# Llama index launcher script


# Kill any existing processes and free up port 4600
lsof -ti :4600 | xargs -r kill -9

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

LOGFILE=/root/logs/llamaindex
mkdir -p "$LOGFILE"

# export CUDA_VISIBLE_DEVICES=1
python -m uvicorn main:app \
	--host 0.0.0.0 \
	--port 4600 \
	--timeout-keep-alive 120 \
	--workers 1  \
	2>&1 | multilog t s5000000 n3 "$LOGFILE" &