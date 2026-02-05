from threading import Lock
from cachetools import LRUCache

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams



Settings.embed_model = HuggingFaceEmbedding(
  model_name="BAAI/bge-m3"
)

class IndexRegistry:
  def __init__(
    self,
    dim: int = 1024,
    max_cache_size: int = 5
  ):
    self.dim = dim
    self._lock = Lock()

    self._collections = LRUCache(maxsize=max_cache_size)

  def get(self, collection_name: str) -> VectorStoreIndex:
    key = collection_name

    with self._lock:        
      if key not in self._collections:
        client = QdrantClient(host="localhost", port=6333)
        if not client.collection_exists(collection_name=collection_name):
          client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
              size=self.dim,
              distance=Distance.COSINE
            )
          )
        vector_store = QdrantVectorStore(
          client=client,
          collection_name=collection_name,
        )
        
        index = VectorStoreIndex.from_vector_store(vector_store)

        self._collections[key] = index

      return self._collections[key]