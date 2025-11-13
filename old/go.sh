export PYTHONPATH="$(pwd)"


python scripts/save_embeddings.py \
  --image ./fordcasepage3.png \
  --output photo_cache.pt \
  --system-prompt "You are analyzing this image."
