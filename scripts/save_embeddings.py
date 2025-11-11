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
    image = Image.open(args.image).convert('RGB')
    
    # Process image and text inputs through the processor
    # The processor handles the conversion to model inputs
    inputs = processor(
        text=args.text,
        images=image,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    logger.info('Extracting embeddings from model...')
    with torch.no_grad():
        # Forward pass through the model to get hidden states
        # We'll use the model's encoder or get embeddings from input stage
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        # Get the last hidden state which represents the processed embeddings
        # This contains both image and text information fused together
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # Use the last hidden state from the model
            inputs_embeds = outputs.hidden_states[-1]
        elif hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states:
            inputs_embeds = outputs.encoder_hidden_states[-1]
        else:
            # Fallback: get embeddings from the embedding layer
            inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
        
        logger.info('Extracted embeddings shape: %s', inputs_embeds.shape)

    # Save embeddings, text, and necessary metadata for inference
    logger.info('Saving embeddings to: %s', args.output)
    save_dict = {
        'text': args.text,
        'inputs_embeds': inputs_embeds.cpu(),
        'model_path': model_path
    }
    
    # Save optional input components that may be needed for generation
    if 'attention_mask' in inputs:
        save_dict['attention_mask'] = inputs['attention_mask'].cpu()
    if 'image_grid_thw' in inputs:
        save_dict['image_grid_thw'] = inputs['image_grid_thw'].cpu()
    if 'input_ids' in inputs:
        save_dict['input_ids'] = inputs['input_ids'].cpu()
    
    torch.save(save_dict, args.output)
    logger.info('Done! Embeddings saved with shape: %s', inputs_embeds.shape)
    logger.info('Use run_inference.py with this file to generate outputs')


if __name__ == "__main__":
    main()
