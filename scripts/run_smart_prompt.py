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

    # Load cached state if provided
    past_key_values = None
    cached_input_ids = None
    cached_attention_mask = None
    cache_info = ""
    
    if args.cache:
        logger.info('Loading cached model state from: %s', args.cache)
        try:
            cache_data = torch.load(args.cache, map_location=device)
            
            past_key_values = cache_data.get('past_key_values')
            cached_input_ids = cache_data.get('input_ids')
            cached_attention_mask = cache_data.get('attention_mask')
            
            cached_model = cache_data.get('model_name', 'unknown')
            cached_system = cache_data.get('system_prompt', '')
            cached_image = cache_data.get('image_path', 'unknown')
            
            if past_key_values:
                # Move KV cache to device
                past_key_values = tuple(
                    tuple(t.to(device) if hasattr(t, 'to') else t for t in layer)
                    for layer in past_key_values
                )
                
            if cached_input_ids is not None:
                cached_input_ids = cached_input_ids.to(device)
            if cached_attention_mask is not None:
                cached_attention_mask = cached_attention_mask.to(device)
            
            logger.info('Loaded cache for model: %s', cached_model)
            logger.info('Cached image: %s', cached_image)
            logger.info('System prompt: %s', cached_system)
            logger.info('KV cache layers: %d', len(past_key_values) if past_key_values else 0)
            logger.info('Cached input length: %d tokens', cached_input_ids.shape[1] if cached_input_ids is not None else 0)
            
            cache_info = f"[Using cached image: {os.path.basename(cached_image)}]"
            
        except Exception as e:
            logger.error('Failed to load cache file.')
            logger.exception(e)
            raise

    # Process the prompt using chat format
    logger.info('Processing prompt: %s', args.prompt)
    try:
        if past_key_values is not None:
            # We have cached state - just tokenize the new prompt
            # Create a follow-up message
            messages = [
                {"role": "user", "content": args.prompt}
            ]
            
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize just the new prompt
            new_inputs = processor(
                text=text,
                return_tensors='pt'
            )
            new_inputs = {k: v.to(device) for k, v in new_inputs.items()}
            
            # Concatenate with cached inputs
            text_inputs = {
                'input_ids': torch.cat([cached_input_ids, new_inputs['input_ids']], dim=1),
                'attention_mask': torch.cat([cached_attention_mask, new_inputs['attention_mask']], dim=1)
            }
            
            logger.info('Using cached state + new prompt (%d cached + %d new tokens)',
                       cached_input_ids.shape[1], new_inputs['input_ids'].shape[1])
        else:
            # No cache - process normally
            messages = [
                {"role": "user", "content": args.prompt}
            ]
            
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            text_inputs = processor(
                text=text,
                return_tensors='pt'
            )
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
            
            # Add cached KV if available
            if past_key_values is not None:
                generate_kwargs['past_key_values'] = past_key_values
                logger.info('Generating with KV cache (%d layers)', len(past_key_values))
            
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
