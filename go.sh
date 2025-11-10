export PYTHONPATH="$(pwd)"

python scripts/save_embeddings.py \
  --text "Create a 3-8 word semantic summary of each paragraph in the image" \
  --image "/workspace/ai-create-embeddings/fordcasepage3.png" \
  --output out.pt \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --verbose
