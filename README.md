# ai-createembeddings

This repository contains a Python library for producing multimodal embeddings using PyTorch and local Qwen3V models. It supports joint text-image embeddings using locally stored GGUF or other model formats.

## Quick Start

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv;source .venv/bin/activate;pip install -r requirements.txt
```

pip cache purge

2. Ensure you have a local Qwen3V model file. The default path is:

```
/models/unsloth_Qwen3-VL-8B-Instruct-GGUF_Qwen3-VL-8B-Instruct-Q4_K_M.gguf
```

3. Run tests with your local model:

```bash
pytest -q
```

## Multimodal Embeddings

The library creates joint text-image embeddings using a local Qwen3V model.

### Basic Usage

```bash
# Save embeddings for later reuse
python scripts/save_embeddings.py \
  --image path/to/image.jpg \
  --text "Describe this landscape" \
  --output embeddings.pt \
  --verbose

# Run inference with saved embeddings (faster, no image processing)
python scripts/run_inference.py \
  --embeddings embeddings.pt \
  --prompt "What colors are in this image?"

# Run multiple prompts using cached embeddings
python scripts/run_inference.py \
  --embeddings embeddings.pt \
  --prompt "Describe the lighting in this image"

python scripts/run_inference.py \
  --embeddings embeddings.pt \
  --prompt "What objects do you see?"

# Specify model path and device
python scripts/save_embeddings.py \
  --image image.jpg \
  --text "A description" \
  --output embeddings.pt \
  --model "/path/to/model.gguf" \
  --device cuda:0
```

### Embedding Caching

The library supports caching image embeddings for faster repeated inference:

1. Save embeddings (`.pt` format):
   - Extracts and saves image embeddings (one-time cost)
   - Stores original text prompt
   - Embeddings saved in CPU format for portability
2. Run inference:
   - Loads cached embeddings directly (very fast)
   - No need to reprocess the image
   - Can run multiple prompts quickly
   - Supports appending to original text

#### Performance Notes

- Initial image embedding extraction has some overhead (loading model, processing image)
- Cached embeddings load instantly - just a tensor load operation
- Image embeddings don't consume context window space
- Using cached embeddings is as fast as text-only inference
- Memory usage is minimal - typically <100MB per cached embedding file

### Model Support

The library works with local Qwen3V models:

- Supports GGUF format (recommended for efficiency)
- Compatible with .bin model files
- Can load from model directories
- Default path: `/models/unsloth_Qwen3-VL-8B-Instruct-GGUF_Qwen3-VL-8B-Instruct-Q4_K_M.gguf`

### Programming Interface

```python
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Extract and save embeddings
def save_embeddings(image_path, text, output_path, model_path=None, device='cuda'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    # Process image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        image_outputs = model.get_image_features(**{k: v for k, v in inputs.items() if k != 'input_ids'})
        image_embeds = image_outputs.image_embeds

    # Save
    torch.save({
        'text': text,
        'image_embeds': image_embeds.cpu()
    }, output_path)

# Load and run inference
def run_inference(embeddings_path, prompt=None, model_path=None, device='cuda'):
    # Load saved data
    data = torch.load(embeddings_path, map_location='cpu')
    image_embeds = data['image_embeds'].to(device)
    text = data['text']

    if prompt:
        text = text + '\n' + prompt

    # Process text and generate
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    text_inputs = processor(text=text, return_tensors='pt')
    inputs = {
        'input_ids': text_inputs['input_ids'].to(device),
        'attention_mask': text_inputs.get('attention_mask', None),
        'image_embeds': image_embeds
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)
        return processor.decode(output_ids[0], skip_special_tokens=True)
```

## Notes

- Requires a local Qwen3V model file (GGUF format recommended)

modelscope download --model ggml-org/Qwen3-VL-2B-Instruct-GGUF README.md --local_dir /models

- Environment variable `TRANSFORMERS_CACHE` can be set to control model loading cache location
- Model files are loaded in local-only mode with trusted code execution
- Use the `--verbose` flag for detailed logging of model loading, processing steps, and embedding shapes
- All embeddings are normalized and can be used directly for similarity computations
- For better performance, consider using quantized models (Q4_K_M or similar)

Do these image embeddings use up context space?

No, the image embeddings don't use up the model's context window space (token limit) in the way text tokens do. Here's why:

1. The `image_embeds` are processed visual features that exist in a separate space from the text context window. They are:

   - Pre-computed visual features from the vision encoder
   - Stored as a fixed-size tensor (typically shape [1, num_visual_features, embedding_dim])
   - Passed directly to the model's cross-attention layers

2. In the code, you can see this separation:

```python
inputs = {
    'input_ids': text_inputs['input_ids'],        # Text tokens (uses context window)
    'attention_mask': text_inputs.get('attention_mask'),  # Mask for text tokens
    'image_embeds': image_embeds                  # Visual features (separate)
}
```

The only things that count toward the context window are:

- The original text (`text`)
- The additional prompt if provided (`args.prompt`)
- The generated tokens (`max_new_tokens`)

This is one of the advantages of using cached image embeddings - they're already in the format the model needs for cross-attention, without taking up any of your text context window. You can use long prompts or generate longer outputs without worrying about the image taking up token space.

The model processes these in parallel:

- Text goes through the text encoder -> text embeddings
- Image features (already encoded) are ready for cross-attention
- The decoder can then attend to both without them competing for context space
