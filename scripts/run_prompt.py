#!/usr/bin/env python3
"""Run a basic AI prompt using the Qwen3-VL-2B-Instruct model.

This script loads the Qwen3V model and processor, then runs a user-provided
prompt and generates text output using PyTorch.

Usage:
  python scripts/run_prompt.py --prompt "What is artificial intelligence?" --max_new_tokens 256

Options:
  --prompt           The text prompt to send to the model (required)
  --model            Model identifier (default: 'Qwen/Qwen3-VL-2B-Instruct')
  --max_new_tokens   Number of tokens to generate (default: 128)
  --device           Device to run on (cpu or cuda). Defaults to cuda if available.
"""
import argparse
import logging
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Run a basic AI prompt on Qwen3-VL-2B-Instruct model'
    )
    parser.add_argument(
        '--prompt',
        required=True,
        help='The text prompt to send to the model'
    )
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-VL-2B-Instruct',
        help='Model identifier (default: Qwen/Qwen3-VL-2B-Instruct)'
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
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
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

    # Process the prompt
    logger.info('Processing prompt: %s', args.prompt)
    try:
        text_inputs = processor(
            text=args.prompt,
            return_tensors='pt'
        )
        # Move inputs to the correct device
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    except Exception as e:
        logger.error('Failed to process prompt.')
        logger.exception(e)
        raise

    # Generate output
    logger.info('Generating text with max_new_tokens=%d', args.max_new_tokens)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **text_inputs,
               # max_new_tokens=args.max_new_tokens,
                #do_sample=True,
                #temperature=0.7,
                #top_p=0.9
            )
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
            output_text = tokenizer.decode(
                output_ids[0],
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
