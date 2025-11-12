  export PYTHONPATH="$(pwd)"

  python scripts/run_prompt.py \
    --prompt "What is 2 + 2?" \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
