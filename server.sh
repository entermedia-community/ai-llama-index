#!/bin/bash
lsof -ti :4600 | xargs -r kill -9

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# export CUDA_VISIBLE_DEVICES=1
python -m uvicorn main:app \
	--host 0.0.0.0 \
	--port 4600 \
	--timeout-keep-alive 120 \
	--workers 1 > /dev/null 2>&1 &