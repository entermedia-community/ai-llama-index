#!/usr/bin/env python3
"""Cache full model state after processing an image.

This script processes an image through the Qwen3V model and caches the complete
model state (past_key_values, hidden states, etc.) so you can prompt multiple times
without reprocessing the image.

Usage:
  python scripts/save_embeddings.py --image image.jpg --output cache.pt
  
The cached state can be loaded by run_smart_prompt.py for fast repeated prompting.
"""
import argparse
import logging
import os
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--output', required=True, help='Path to save cache .pt file')
    parser.add_argument('--system-prompt', default='You are a helpful AI assistant analyzing an image.', 
                       help='System prompt for the image context')
    parser.add_argument('--model', required=False, default=None, help='Local model path')
    parser.add_argument('--device', required=False, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    log_level = logging.DEBUG # if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger('save_embeddings')

    if not os.path.exists(args.image):
        logger.error('Image file not found: %s', args.image)
        raise SystemExit(1)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Running on device: %s', device)

    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except Exception as e:
        logger.exception('Failed to import Qwen3V classes from transformers')
        raise

    model_path = args.model or 'Qwen/Qwen3-VL-2B-Instruct'
    # if not os.path.exists(model_path):
    #     logger.error('Model path does not exist: %s', model_path)
    #     raise SystemExit(1)
    print("Model:", model_path)
    
    logger.info('Loading processor and model from: %s', model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, 
        dtype="auto",
        device_map="auto",
    )

    # Load and process image with system prompt
    logger.info('Processing image: %s', args.image)
    
    # Create initial conversation with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.system_prompt}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    logger.info('Processing image and system prompt through model...')
    # Process everything together
    inputs = processor(
        text=[text],
        images=[args.image],
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run a forward pass to build the KV cache with the image processed
    logger.info('Running forward pass to cache image processing...')
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True
        )
        
        # Extract the cached state
        past_key_values = outputs.past_key_values
        last_hidden_state = outputs.logits  # Last token logits
    
    # Prepare cache data
    cache_data = {
        'model_name': model_path,
        'system_prompt': args.system_prompt,
        'image_path': args.image,
        'past_key_values': past_key_values,
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'processor_config': {
            'chat_template': processor.chat_template if hasattr(processor, 'chat_template') else None
        }
    }
    
    # Save the cache
    logger.info('Saving model cache to: %s', args.output)
    torch.save(cache_data, args.output)
    
    # Calculate cache size
    cache_size_mb = os.path.getsize(args.output) / (1024 * 1024) if os.path.exists(args.output) else 0
    logger.info('Cache saved successfully (%.2f MB)', cache_size_mb)
    logger.info('KV cache layers: %d', len(past_key_values) if past_key_values else 0)
    logger.info('Input tokens: %d', inputs['input_ids'].shape[1])
    logger.info('Use run_smart_prompt.py with --cache %s to prompt without reprocessing', args.output)

if __name__ == "__main__":
    main()
