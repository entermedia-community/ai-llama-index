#!/usr/bin/env python3
"""Run inference using saved image embeddings with the Qwen3V model.

This script expects a .pt file containing saved image embeddings and original text.
It will load the embeddings directly without needing the original image file.

Usage:
  python scripts/run_inference.py --embeddings embeddings.pt --model /path/to/qwen3v.gguf --device cpu

Options:
  --embeddings   Path to the .pt file saved by save_embeddings.py
  --model        Local model path (directory or file) to load Qwen3V from. If omitted,
                 the default in `src/embedder/multimodal.py` is used.
  --device       Device to run on (cpu or cuda). Defaults to 'cuda' if available.
  --prompt       Optional extra prompt / question to append to the saved text
  --max_new_tokens Number of tokens to generate (default: 128)
"""
import argparse
import logging
import os
import torch
# from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True, help='Path to .pt embeddings file')
    parser.add_argument('--model', required=False, default=None, help='Local model path to load (overrides default)')
    # parser.add_argument('--device', required=False, default=None, help='Device to run on (cpu/cuda)')
    parser.add_argument('--prompt', required=False, default=None, help='Optional extra text to append to the saved text')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('run_inference')

    if not os.path.exists(args.embeddings):
        logger.error('Embeddings file not found: %s', args.embeddings)
        raise SystemExit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Running on device: %s', device)

    # Load embeddings file
    logger.info('Loading embeddings from: %s', args.embeddings)
    data = torch.load(args.embeddings, map_location='cpu')

    text = data.get('text')
    if not text:
        logger.error('No text found in embeddings file')
        raise SystemExit(1)

    # Import transformers here (fail early with clear message)
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except Exception as e:
        logger.exception('Failed to import Qwen3V classes from transformers; ensure your venv has a transformers build with Qwen3V support')
        raise

    # Use the model path from saved embeddings or the provided argument
    model_path = args.model or data.get('model_path') or 'Qwen/Qwen3-VL-8B-Instruct'

    logger.info('Loading processor and model from: %s', model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto"
    )

    # Get the device from the model's first parameter (works with device_map="auto")
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    logger.info('Model loaded on device: %s with dtype: %s', model_device, model_dtype)

    # Load the saved embeddings
    inputs_embeds = data.get('inputs_embeds')
    if inputs_embeds is None:
        logger.error('No inputs_embeds key found in %s', args.embeddings)
        raise SystemExit(1)
    
    # Move embeddings to model's device and dtype
    inputs_embeds = inputs_embeds.to(model_device).to(model_dtype)
    logger.info('Loaded input embeddings with shape: %s', inputs_embeds.shape)
    
    # Prepare generation kwargs
    gen_kwargs = {
        'inputs_embeds': inputs_embeds,
        'max_new_tokens': args.max_new_tokens,
    }
    
    # Add optional components if they exist
    if 'attention_mask' in data and data['attention_mask'] is not None:
        gen_kwargs['attention_mask'] = data['attention_mask'].to(model_device)
        logger.debug('Using saved attention_mask')
    
    if args.prompt:
        logger.info('Additional prompt provided: %s', args.prompt)
        logger.warning('Note: Adding extra prompt to pre-computed embeddings may not work as expected')

    # Generate using the pre-computed embeddings
    logger.info('Generating with max_new_tokens=%d', args.max_new_tokens)
    with torch.no_grad():
        try:
            # Use inputs_embeds for generation with pre-computed embeddings
            gen = model.generate(**gen_kwargs)
        except TypeError as e:
            logger.error('Model.generate() raised TypeError: %s', e)
            logger.error('This may indicate an incompatible transformers version. Ensure your transformers build supports Qwen3V.')
            raise

    # Decode
    output_text = None
    try:
        # Processor likely exposes a tokenizer
        tokenizer = getattr(processor, 'tokenizer', None)
        if tokenizer is not None:
            output_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        else:
            # try model's tokenizer attr
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is not None:
                output_text = tokenizer.decode(gen[0], skip_special_tokens=True)
    except Exception:
        output_text = None

    if output_text is None:
        # Fallback to string of token ids
        logger.warning('Could not decode tokens to text using processor/tokenizer; printing token ids instead')
        output_text = str(gen[0].tolist())

    print('\n=== Generated Output ===\n')
    print(output_text)


if __name__ == '__main__':
    main()