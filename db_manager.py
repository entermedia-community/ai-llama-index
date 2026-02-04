from threading import Lock
from cachetools import LRUCache
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings, VectorStoreIndex

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
        vector_store = MilvusVectorStore(
          collection_name=key,
          dim=self.dim,
          # uri="http://mediadb45.entermediadb.net:19530",
          uri="mil.db",
          overwrite=True,
        )
        
        index = VectorStoreIndex.from_vector_store(vector_store)

        self._collections[key] = index

      return self._collections[key]