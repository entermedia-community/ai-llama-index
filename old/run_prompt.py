#!/usr/bin/env python3
"""
Run prompts using saved embeddings from llama.cpp.

Usage:
    python run_prompt.py --model /path/to/model.gguf --embeddings embeddings.json --prompt "Your question here"
    python run_prompt.py --model model.gguf --embeddings embeddings.json --prompt "Question" --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from llama_cpp import Llama
import numpy as np


def load_embeddings(embeddings_path: str) -> Dict[str, Any]:
    """Load embeddings from a JSON file."""
    try:
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate the structure
        if 'embeddings' not in data or 'texts' not in data:
            raise ValueError("Invalid embeddings file format")
        
        return data
    except Exception as e:
        print(f"Error loading embeddings: {e}", file=sys.stderr)
        sys.exit(1)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def find_most_relevant(
    prompt_embedding: List[float],
    stored_embeddings: List[List[float]],
    stored_texts: List[str],
    top_k: int = 3
) -> List[tuple]:
    """
    Find the most relevant stored texts based on embedding similarity.
    
    Returns:
        List of (index, similarity_score, text) tuples
    """
    similarities = []
    for i, stored_emb in enumerate(stored_embeddings):
        sim = cosine_similarity(prompt_embedding, stored_emb)
        similarities.append((i, sim, stored_texts[i]))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def generate_response(
    model_path: str,
    prompt: str,
    context: str = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    n_ctx: int = 2048,
    n_gpu_layers: int = 0,
    verbose: bool = False
) -> str:
    """
    Generate a response using llama.cpp.
    
    Args:
        model_path: Path to the GGUF model file
        prompt: The user's prompt/question
        context: Optional context to prepend to the prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU
        verbose: Enable verbose output
    
    Returns:
        Generated text response
    """
    if verbose:
        print(f"Loading model: {model_path}")
        print(f"Context size: {n_ctx}")
        print(f"GPU layers: {n_gpu_layers}")
    
    # Initialize llama.cpp model in generation mode
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose
    )
    
    # Build the full prompt with context if provided
    full_prompt = prompt
    if context:
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
    
    if verbose:
        print(f"\nGenerating response...")
        print(f"Prompt length: {len(full_prompt)} chars")
    
    # Generate response
    output = llm(
        full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["Question:", "\n\n"],
        echo=False
    )
    
    return output['choices'][0]['text'].strip()


def main():
    parser = argparse.ArgumentParser(
        description='Run prompts using saved embeddings from llama.cpp',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a prompt with semantic search in embeddings
  python run_prompt.py --model model.gguf --embeddings embeddings.json --prompt "What is this about?"
  
  # Use GPU acceleration and adjust parameters
  python run_prompt.py --model model.gguf --embeddings data.json --prompt "Question?" --gpu-layers 32 --max-tokens 512
  
  # Show top 5 most relevant contexts
  python run_prompt.py --model model.gguf --embeddings data.json --prompt "Query" --top-k 5 --verbose
  
  # Direct generation without semantic search
  python run_prompt.py --model model.gguf --embeddings data.json --prompt "Question" --no-search
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the GGUF model file'
    )
    
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to the saved embeddings JSON file'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Your question or prompt'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate (default: 256)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    
    parser.add_argument(
        '--context-size',
        type=int,
        default=2048,
        help='Context window size (default: 2048)'
    )
    
    parser.add_argument(
        '--gpu-layers',
        type=int,
        default=0,
        help='Number of layers to offload to GPU (0 = CPU only, default: 0)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of most relevant contexts to include (default: 3)'
    )
    
    parser.add_argument(
        '--no-search',
        action='store_true',
        help='Skip semantic search and just generate a response'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.embeddings).exists():
        print(f"Error: Embeddings file not found: {args.embeddings}", file=sys.stderr)
        sys.exit(1)
    
    # Load saved embeddings
    if args.verbose:
        print(f"Loading embeddings from {args.embeddings}...")
    
    data = load_embeddings(args.embeddings)
    
    if args.verbose:
        print(f"Loaded {data['num_embeddings']} embeddings")
        print(f"Embedding dimension: {data['embedding_dim']}")
    
    context = None
    
    if not args.no_search:
        # Create embedding for the prompt using the same model
        if args.verbose:
            print(f"\nCreating embedding for prompt...")
        
        llm_embed = Llama(
            model_path=args.model,
            n_ctx=512,
            n_gpu_layers=args.gpu_layers,
            embedding=True,
            verbose=False
        )
        
        prompt_embedding = llm_embed.embed(args.prompt)
        
        if args.verbose:
            print(f"Finding most relevant contexts (top {args.top_k})...")
        
        # Find most relevant stored texts
        relevant = find_most_relevant(
            prompt_embedding,
            data['embeddings'],
            data['texts'],
            top_k=args.top_k
        )
        
        # Display relevance scores
        if args.verbose:
            print("\nMost relevant contexts:")
            for i, (idx, score, text) in enumerate(relevant, 1):
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"  {i}. Score: {score:.4f} - {preview}")
        
        # Combine top contexts
        context_parts = []
        for idx, score, text in relevant:
            if score > 0.1:  # Only include if somewhat relevant
                context_parts.append(text)
        
        if context_parts:
            context = "\n\n".join(context_parts)
    
    # Generate response
    try:
        response = generate_response(
            model_path=args.model,
            prompt=args.prompt,
            context=context,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            n_ctx=args.context_size,
            n_gpu_layers=args.gpu_layers,
            verbose=args.verbose
        )
        
        print("\n" + "="*60)
        print("RESPONSE:")
        print("="*60)
        print(response)
        print("="*60)
        
    except Exception as e:
        print(f"Error generating response: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
