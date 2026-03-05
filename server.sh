export CUDA_VISIBLE_DEVICES=1
uvicorn main:app --port 9600 > /dev/null 2>&1 &