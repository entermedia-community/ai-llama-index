# Model State Caching Usage Guide

This guide explains how to use the model state caching feature to avoid reprocessing images when prompting multiple times.

## Overview

The caching system processes an image once through the vision encoder and saves the complete model state (KV cache, hidden states, etc.). You can then prompt multiple times without reprocessing the image, saving significant computation time.

## Quick Start

### Step 1: Process and Cache an Image

```bash
python scripts/save_embeddings.py \
  --image /path/to/photo.jpg \
  --output photo_cache.pt \
  --system-prompt "You are analyzing this image."
```

**Options:**
- `--image` (required): Path to the image file
- `--output` (required): Where to save the cache file (.pt)
- `--system-prompt`: Initial context for the image (default: "You are a helpful AI assistant analyzing an image.")
- `--model`: Model path (default: Qwen/Qwen3-VL-2B-Instruct)
- `--device`: cpu or cuda (auto-detected by default)

### Step 2: Prompt Using the Cache

```bash
# First prompt
python scripts/run_smart_prompt.py \
  --cache photo_cache.pt \
  --prompt "What objects are visible in this image?" \
  --max_new_tokens 256

# Second prompt (still using same cache)
python scripts/run_smart_prompt.py \
  --cache photo_cache.pt \
  --prompt "What colors dominate the scene?" \
  --max_new_tokens 128

# Third prompt
python scripts/run_smart_prompt.py \
  --cache photo_cache.pt \
  --prompt "Describe the lighting and mood." \
  --max_new_tokens 200
```

**Options:**
- `--cache`: Path to cached model state (from save_embeddings.py)
- `--prompt` (required): Your question/prompt
- `--max_new_tokens`: How many tokens to generate (default: 128)
- `--model`: Model path (should match the cache)
- `--device`: cpu or cuda

## What Gets Cached?

The cache file contains:
- **KV cache** (past_key_values): Attention states from processing the image
- **Input tokens**: The tokenized image + system prompt
- **Attention masks**: Required for continuing generation
- **Metadata**: Model name, system prompt, image path

## Performance Benefits

### Without Cache (normal mode):
- Load model: ~5-10s
- Process image through vision encoder: ~2-5s per prompt
- Generate text: ~1-10s
- **Total per prompt: ~8-25s**

### With Cache:
- Load model: ~5-10s (once)
- Load cache: ~0.5-2s
- Process new prompt: ~0.1s
- Generate text: ~1-10s
- **Total per prompt: ~1.6-12s**

**Speedup: 2-5x faster for repeated prompting on the same image**

## File Sizes

Cache files typically range from:
- **50-200 MB** for 2B parameter models
- **200-500 MB** for 8B parameter models

The size depends on:
- Model size
- Number of attention layers
- Image resolution
- Length of system prompt

## Advanced Usage

### Custom System Prompts

```bash
# Art analysis assistant
python scripts/save_embeddings.py \
  --image artwork.jpg \
  --output art_cache.pt \
  --system-prompt "You are an expert art historian analyzing this artwork. Focus on composition, technique, and historical context."

# Medical imaging assistant  
python scripts/save_embeddings.py \
  --image xray.jpg \
  --output medical_cache.pt \
  --system-prompt "You are a radiologist assistant analyzing this medical image. Be precise and technical."
```

### Batch Processing Multiple Images

```bash
# Cache multiple images
for img in images/*.jpg; do
  basename=$(basename "$img" .jpg)
  python scripts/save_embeddings.py \
    --image "$img" \
    --output "caches/${basename}_cache.pt"
done

# Query each cache
for cache in caches/*_cache.pt; do
  echo "Processing $cache"
  python scripts/run_smart_prompt.py \
    --cache "$cache" \
    --prompt "Describe this image in one sentence."
done
```

## Without Cache (Fallback)

You can still use `run_smart_prompt.py` without a cache for regular text-only prompts:

```bash
python scripts/run_smart_prompt.py \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 256
```

## Troubleshooting

### "Model mismatch" error
- Ensure you're using the same model for caching and prompting
- Check the `--model` argument matches in both scripts

### "Out of memory" error
- Use `--device cpu` for caching if GPU memory is limited
- Reduce image resolution before caching
- Close other GPU applications

### Cache file too large
- Use a smaller model (2B instead of 8B)
- Shorten the system prompt
- Reduce image resolution

### Slow generation even with cache
- Check that you're actually using `--cache` argument
- Ensure cache file loaded successfully (check logs)
- GPU memory might be fragmented - restart script

## Technical Details

The caching works by:
1. Processing the image through the vision encoder (expensive)
2. Running a forward pass with the system prompt
3. Saving the KV cache (attention states) at that point
4. When prompting later, we append new tokens and continue from that state
5. The model doesn't need to recompute vision features

This is similar to how ChatGPT remembers conversation history - we're essentially "pre-loading" the image into the conversation.
