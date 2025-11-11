#!/usr/bin/env python3
"""Save image embeddings for later inference.

This script processes an image through the Qwen3V model and saves its embeddings
along with the original text prompt. The embeddings can be loaded by run_inference.py
for faster repeated inference without needing to reprocess the image.

Usage:
  python scripts/save_embeddings.py --image image.jpg --text "Description" --output embeddings.pt
"""
import argparse
import logging
import os
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--text', required=True, help='Text prompt/description')
    parser.add_argument('--output', required=True, help='Path to save embeddings .pt file')
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

    model_path = args.model or 'Qwen/Qwen3-VL-8B-Instruct'
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

    # Load and process image
    logger.info('Processing image: %s', args.image)
    image = Image.open(args.image) #.convert('RGB')
    
    # Get image embeddings
    image_inputs = processor(images=[args.image], text=[args.text], return_tensors='pt')

    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    
    image_embeds = None

    with torch.no_grad():
        # Extract image features/embeddings
        image_embeds = model.get_image_features(**{k: v for k, v in image_inputs.items() if k != 'input_ids' and k != 'attention_mask'})
        #image_embeds = image_outputs.image_embeds

    # Save embeddings and text
    logger.info('Saving embeddings to: %s', args.output)
    torch.save({
        'text': args.text,
        'image_embeds': image_embeds
    }, args.output)
    logger.info('Done! Use run_inference.py with this file to generate outputs')


if __name__ == "__main__":
    main()
