#!/usr/bin/env python3
"""
Create embeddings from text files using llama.cpp and save them for later use.

Usage:
    python create_embeddings.py --model /path/to/model.gguf --input text.txt --output embeddings.json
    python create_embeddings.py --model /path/to/model.gguf --text "Direct text input" --output embeddings.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from llama_cpp import Llama


def load_text_from_file(file_path: str) -> str:
    """Load text content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def create_embeddings(
    model_path: str,
    texts: List[str],
    n_ctx: int = 512,
    n_gpu_layers: int = 0,
    verbose: bool = False
) -> List[List[float]]:
    """
    Create embeddings using llama.cpp.
    
    Args:
        model_path: Path to the GGUF model file
        texts: List of text strings to embed
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
        verbose: Enable verbose output
    
    Returns:
        List of embedding vectors (one per text)
    """
    if verbose:
        print(f"Loading model: {model_path}")
        print(f"Context size: {n_ctx}")
        print(f"GPU layers: {n_gpu_layers}")
    
    # Initialize llama.cpp model with embedding support
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        embedding=True,  # Enable embedding mode
        verbose=verbose
    )
    
    embeddings = []
    for i, text in enumerate(texts):
        if verbose:
            print(f"Processing text {i+1}/{len(texts)} ({len(text)} chars)...")
        
        # Get embedding for this text
        embedding = llm.embed(text)
        embeddings.append(embedding)
        
        if verbose:
            print(f"  Embedding dimension: {len(embedding)}")
    
    return embeddings


def save_embeddings(
    embeddings: List[List[float]],
    texts: List[str],
    output_path: str,
    metadata: Dict[str, Any] = None
) -> None:
    """Save embeddings to a JSON file with metadata."""
    data = {
        'embeddings': embeddings,
        'texts': texts,
        'metadata': metadata or {},
        'version': '1.0',
        'embedding_dim': len(embeddings[0]) if embeddings else 0,
        'num_embeddings': len(embeddings)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(embeddings)} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create embeddings from text using llama.cpp',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create embeddings from a text file
  python create_embeddings.py --model model.gguf --input document.txt --output embeddings.json
  
  # Create embeddings from direct text input
  python create_embeddings.py --model model.gguf --text "Hello world" --output embeddings.json
  
  # Use GPU acceleration
  python create_embeddings.py --model model.gguf --input text.txt --output out.json --gpu-layers 32
  
  # Process multiple files
  python create_embeddings.py --model model.gguf --input file1.txt file2.txt --output embeddings.json
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the GGUF model file (e.g., llama-2-7b.Q4_K_M.gguf)'
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=str,
        nargs='+',
        help='Path to input text file(s) to create embeddings from'
    )
    input_group.add_argument(
        '--text',
        type=str,
        help='Direct text input to create embeddings from'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save the embeddings JSON file'
    )
    
    parser.add_argument(
        '--context-size',
        type=int,
        default=512,
        help='Context window size (default: 512)'
    )
    
    parser.add_argument(
        '--gpu-layers',
        type=int,
        default=0,
        help='Number of layers to offload to GPU (0 = CPU only, default: 0)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    
    # Collect texts to embed
    texts = []
    if args.input:
        for input_file in args.input:
            if not Path(input_file).exists():
                print(f"Error: Input file not found: {input_file}", file=sys.stderr)
                sys.exit(1)
            texts.append(load_text_from_file(input_file))
    else:
        texts.append(args.text)
    
    if args.verbose:
        print(f"Creating embeddings for {len(texts)} text(s)...")
    
    # Create embeddings
    try:
        embeddings = create_embeddings(
            model_path=args.model,
            texts=texts,
            n_ctx=args.context_size,
            n_gpu_layers=args.gpu_layers,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error creating embeddings: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save to file
    metadata = {
        'model': args.model,
        'context_size': args.context_size,
        'gpu_layers': args.gpu_layers,
        'input_files': args.input if args.input else None,
    }
    
    save_embeddings(embeddings, texts, args.output, metadata)
    
    if args.verbose:
        print("Done!")


if __name__ == '__main__':
    main()
