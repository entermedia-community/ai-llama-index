import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import asyncio
from functools import partial
from typing import Optional
import logging

from fastapi import Header, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)

llm = OpenAILike(
    #api_base="http://vast02:42063/", # Server uses local LLM
    api_base="https://llamat.emediaworkspace.com/", # Use this for testing locally with the remote LLM
    api_key="dummy_api_key",  # Not used for local LLM, but required by the OpenAILike class
    is_chat_model=True,
    is_function_calling_model=True
)

Settings.llm = llm

Settings.embed_model = HuggingFaceEmbedding(
  model_name="BAAI/bge-m3"
)

client = QdrantClient(
    # host="localhost",
    # port=6333,
    host="0.0.0.0", 
    port=6333
)

aclient = AsyncQdrantClient(
    # host="localhost",
    # port=6333,
    host="0.0.0.0", 
    port=6333
)

VECTOR_SIZE = 1024

def ensure_collection(name: str):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

# @lru_cache(maxsize=32)
def get_vector_store(collection: str) -> QdrantVectorStore:
    ensure_collection(collection)
    return QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=collection,
        prefer_grpc=True,
    )

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
INDEX_TIMEOUT_SECONDS = int(os.getenv("INDEX_TIMEOUT_SECONDS", "30"))
MAX_CONCURRENT_HEAVY_REQUESTS = int(os.getenv("MAX_CONCURRENT_HEAVY_REQUESTS", "4"))
heavy_request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_HEAVY_REQUESTS)


async def run_blocking(func, *args, timeout: int = REQUEST_TIMEOUT_SECONDS, **kwargs):
    """Run sync heavy tasks in a threadpool to keep the event loop responsive."""
    bound_call = partial(func, *args, **kwargs)
    return await asyncio.wait_for(run_in_threadpool(bound_call), timeout=timeout)

def get_collection_name(x_customerkey: Optional[str] = Header(None)):
    if not x_customerkey or not x_customerkey.isalnum():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid customer key.")
    return f'client_{x_customerkey}_embeddings'
