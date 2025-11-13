#!/usr/bin/env python3
"""Simple CLI for producing embeddings from stdin or args."""
import sys
from argparse import ArgumentParser
from src.embedder.model import EmbeddingModel


def main():
    p = ArgumentParser()
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("text", nargs="*", help="Text(s) to embed. If omitted, reads from stdin.")
    args = p.parse_args()

    if args.text:
        texts = args.text
    else:
        texts = [line.strip() for line in sys.stdin if line.strip()]

    m = EmbeddingModel(model_name=args.model)
    embs = m.embed(texts)

    for t, e in zip(texts, embs):
        print(t)
        print(e)


if __name__ == "__main__":
    main()
