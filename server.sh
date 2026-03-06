export CUDA_VISIBLE_DEVICES=1
uvicorn main:app \
	--host 0.0.0.0 \
	--port 4600 \
	--workers "${UVICORN_WORKERS:-1}" \
	--timeout-keep-alive "${UVICORN_TIMEOUT_KEEP_ALIVE:-120}" \
	--log-level "${UVICORN_LOG_LEVEL:-info}" \
	> uvicorn.log 2>&1 &