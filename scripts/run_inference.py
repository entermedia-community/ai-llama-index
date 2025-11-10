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

    device = "cuda" # args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Running on device: %s', device)

    # Load embeddings file
    data = torch.load(args.embeddings)

    text = data.get('text')

    if args.prompt:
        prompt_text = text + '\n' + args.prompt
    else:
        prompt_text = text

    # Import transformers here (fail early with clear message)
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except Exception as e:
        logger.exception('Failed to import Qwen3V classes from transformers; ensure your venv has a transformers build with Qwen3V support')
        raise

    model_path = args.model or 'Qwen/Qwen3-VL-8B-Instruct'

    # if not os.path.exists(model_path):
    #     logger.error('Model path does not exist: %s', model_path)
    #     raise SystemExit(1)

    logger.info('Loading processor and model from: %s', model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto"
    )

    image_embeds = data.get('image_embeds')
    # image_embeds = image_embeds.to(model.device).to(model.dtype)

    # Move image embeddings to device
    # image_embeds = image_embeds.to(device)
    
    # Process text only
    text_inputs = processor(text=prompt_text, return_tensors='pt')
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # text_embeds = model.get_text_features(text_inputs['input_ids'])
    # inputs_embeds = torch.cat([text_embeds, image_embeds], dim=1)
    # Generate
    logger.info('Generating with max_new_tokens=%d', args.max_new_tokens)
    with torch.no_grad():
        try:
            gen = model.generate(
                # input_ids=text_inputs['input_ids'],
                # attention_mask=text_inputs['attention_mask'],
                inputs_embeds=image_embeds,
                # attention_mask=torch.ones(image_embeds.shape[:2], device=model.device),
                max_new_tokens=100
            )
        except TypeError:
            logger.error('Model.generate() raised TypeError; this may indicate an incompatible transformers version. Ensure your transformers build supports Qwen3V.')
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