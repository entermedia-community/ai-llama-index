#!/usr/bin/env python3
"""Run prompts using cached model state from image processing.

This script loads a cached model state (from save_embeddings.py) and runs
new prompts without reprocessing the image, saving significant computation.

Usage:
  python scripts/run_smart_prompt.py --cache cache.pt --prompt "What colors do you see?"
  
Or without cache (processes from scratch):
  python scripts/run_smart_prompt.py --prompt "What is AI?" --max_new_tokens 256

Options:
  --prompt           The text prompt to send to the model (required)
  --cache            Path to cached model state from save_embeddings.py
  --model            Model identifier (default: 'Qwen/Qwen3-VL-8B-Instruct')
  --max_new_tokens   Number of tokens to generate (default: 128)
  --device           Device to run on (cpu or cuda). Defaults to cuda if available.
"""
import argparse
import logging
import os
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Run a basic AI prompt on Qwen3-VL-8B-Instruct model'
    )
    parser.add_argument(
        '--prompt',
        required=True,
        help='The text prompt to send to the model'
    )
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-VL-8B-Instruct',
        help='Model identifier (default: Qwen/Qwen3-VL-8B-Instruct)'
    )
    parser.add_argument(
        '--cache',
        help='Path to cached model state from save_embeddings.py'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=128,
        help='Number of tokens to generate (default: 128)'
    )
    parser.add_argument(
        '--device',
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to run on (default: cuda if available, else cpu)'
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('run_prompt')

    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Using device: %s', device)

    # Import transformers
    try:
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, GenerationConfig
    except Exception as e:
        logger.error(
            'Failed to import Qwen3V classes from transformers. '
            'Ensure your environment has a transformers build with Qwen3V support.'
        )
        logger.exception(e)
        raise

    logger.info('Loading model and processor from: %s', args.model)
    try:
        processor = Qwen3VLProcessor.from_pretrained(args.model)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype='auto',
            device_map='auto'
        )
    except Exception as e:
        logger.error('Failed to load model or processor.')
        logger.exception(e)
        raise

    logger.info('Model loaded successfully')

    # Load processed image inputs if provided
    image_inputs = {}
    cache_info = ""
    
    if args.cache:
        logger.info('Loading processed image from: %s', args.cache)
        try:
            # Use weights_only=False for cache files containing transformers objects
            # This is safe since we created the cache file ourselves
            cache_data = torch.load(args.cache, map_location=device, weights_only=False)
            
            image_inputs = cache_data.get('image_inputs', {})
            cached_image = cache_data.get('image_path', 'unknown')
            
            # Move tensors to device
            image_inputs = {k: v.to(device) if hasattr(v, 'to') else v 
                           for k, v in image_inputs.items()}
            
            logger.info('Loaded image: %s', cached_image)
            logger.info('Image input keys: %s', list(image_inputs.keys()))
            for k, v in image_inputs.items():
                if hasattr(v, 'shape'):
                    logger.info('  %s shape: %s', k, v.shape)
            
            cache_info = f"[Image: {os.path.basename(cached_image)}]"
            
        except Exception as e:
            logger.error('Failed to load cache file.')
            logger.exception(e)
            raise

    # Process the prompt using chat format
    logger.info('Processing prompt: %s', args.prompt)
    try:
        if image_inputs:
            # We have image - create multimodal message
            messages = [
                {
                    "role": "user",
                    "content": args.prompt  # Just text, image will be added separately
                }
            ]
        else:
            # Text-only message
            messages = [
                {"role": "user", "content": args.prompt}
            ]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process text (and image if available)
        if image_inputs:
            # Combine text with cached image inputs
            text_inputs = processor(
                text=text,
                return_tensors='pt'
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            # Merge with image inputs
            text_inputs.update(image_inputs)
            logger.info('Using cached image with new prompt')
        else:
            # Text only
            text_inputs = processor(
                text=text,
                return_tensors='pt'
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    except Exception as e:
        logger.error('Failed to process prompt.')
        logger.exception(e)
        raise
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    except Exception as e:
        logger.error('Failed to process prompt.')
        logger.exception(e)
        raise

    logger.info('Generating text with max_new_tokens=%d', args.max_new_tokens)

    gen_config = GenerationConfig.from_pretrained(args.model)
    gen_config.temperature = 0.5
    gen_config.do_sample = True
    gen_config.max_new_tokens = args.max_new_tokens

    try:
        with torch.no_grad():
            # Prepare generate arguments
            generate_kwargs = {
                **text_inputs,
                'generation_config': gen_config
            }
            
            output_ids = model.generate(**generate_kwargs)
    except Exception as e:
        logger.error('Failed during model.generate().')
        logger.exception(e)
        raise

    # Decode the output
    logger.info('Decoding output tokens')
    try:
        # Try to get tokenizer from processor
        tokenizer = getattr(processor, 'tokenizer', None)
        if tokenizer is None:
            # Try to get tokenizer from model
            tokenizer = getattr(model, 'tokenizer', None)

        if tokenizer is not None:
            # Decode only the generated tokens (skip the input)
            generated_ids = output_ids[0][text_inputs['input_ids'].shape[1]:]
            output_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
        else:
            logger.warning('No tokenizer found; printing token IDs instead')
            output_text = str(output_ids[0].tolist())
    except Exception as e:
        logger.warning('Failed to decode tokens; printing token IDs instead')
        output_text = str(output_ids[0].tolist())

    # Print results
    print('\n' + '='*60)
    if cache_info:
        print(cache_info)
    print('PROMPT')
    print('='*60)
    print(args.prompt)
    print('\n' + '='*60)
    print('GENERATED OUTPUT')
    print('='*60)
    print(output_text)
    print('='*60 + '\n')

    logger.info('Inference completed successfully')


if __name__ == '__main__':
    main()
