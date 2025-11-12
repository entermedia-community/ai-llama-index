#!/usr/bin/env python3
"""Run a basic AI prompt using the Qwen3-VL-8B-Instruct model.

This script loads the Qwen3V model and processor, then runs a user-provided
prompt and generates text output using PyTorch.

Usage:
  python scripts/run_prompt.py --prompt "What is artificial intelligence?" --max_new_tokens 256

Options:
  --prompt           The text prompt to send to the model (required)
  --embeddings       Read in the embeddings from a .pt file saved by save_embeddings.py
  --model            Model identifier (default: 'Qwen/Qwen3-VL-8B-Instruct')
  --max_new_tokens   Number of tokens to generate (default: 128)
  --device           Device to run on (cpu or cuda). Defaults to cuda if available.
"""
import argparse
import logging
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
        '--embeddings',
        help='Read in the embeddings from a .pt file saved by save_embeddings.py'
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

    # Load embeddings if provided
    image_embeds = None
    if args.embeddings:
        logger.info('Loading embeddings from: %s', args.embeddings)
        try:
            embeddings_data = torch.load(args.embeddings, map_location=device)
            raw_embeds = embeddings_data['image_embeds']
            
            # Handle tuple or tensor - recursively extract if needed
            def extract_tensor(obj):
                """Recursively extract the first tensor from nested tuples."""
                if hasattr(obj, 'shape') and hasattr(obj, 'to'):
                    # It's a tensor
                    return obj
                elif isinstance(obj, (tuple, list)) and len(obj) > 0:
                    # It's a tuple/list, try first element
                    return extract_tensor(obj[0])
                else:
                    return obj
            
            image_embeds = extract_tensor(raw_embeds)
            
            # Move to device if it's a tensor
            if hasattr(image_embeds, 'to'):
                image_embeds = image_embeds.to(device)
            
            saved_text = embeddings_data.get('text', '')
            logger.info('Loaded embeddings with saved text: %s', saved_text)
            logger.info('Embeddings type: %s', type(image_embeds))
            if hasattr(image_embeds, 'shape'):
                logger.info('Embeddings shape: %s', image_embeds.shape)
        except Exception as e:
            logger.error('Failed to load embeddings file.')
            logger.exception(e)
            raise

    # Process the prompt using chat format
    logger.info('Processing prompt: %s', args.prompt)
    try:
        # Create messages in chat format
        messages = [
            {"role": "user", "content": args.prompt}
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process the formatted chat text
        text_inputs = processor(
            text=text,
            return_tensors='pt'
        )
        # Move inputs to the correct device
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
            
            # Add image embeddings if available
            if image_embeds is not None:
                embed_shape = image_embeds.shape if hasattr(image_embeds, 'shape') else f'type: {type(image_embeds)}'
                logger.info('Adding image embeddings to generation (shape: %s)', embed_shape)
                
                # If still a tuple, we need to handle it differently
                if isinstance(image_embeds, tuple):
                    logger.warning('image_embeds is still a tuple, attempting to extract tensor')
                    # Try different tuple elements
                    for i, elem in enumerate(image_embeds):
                        logger.info('Tuple element %d: type=%s, shape=%s', i, type(elem), 
                                   getattr(elem, 'shape', 'no shape'))
                    # Use the first tensor we find
                    for elem in image_embeds:
                        if hasattr(elem, 'shape'):
                            image_embeds = elem
                            logger.info('Using tensor with shape: %s', image_embeds.shape)
                            break
                
                generate_kwargs['image_embeds'] = image_embeds
            
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
