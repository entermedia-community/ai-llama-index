  export PYTHONPATH="$(pwd)"

  python scripts/run_smart_prompt.py \
    --embeddings out.pt \
    --prompt "What is in the image?" 
