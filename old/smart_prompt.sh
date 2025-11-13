  export PYTHONPATH="$(pwd)"

  python scripts/run_smart_prompt.py \
    --cache photo_cache.pt \
    --prompt "What is in the image?" 
