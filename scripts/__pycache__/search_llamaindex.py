from llama_index.core import load_index_from_storage

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#https://github.com/QwenLM/Qwen3-Embedding

embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-0.6B")

from llama_index.readers.json import JSONReader

reader = JSONReader()
documents = reader.load_data(input_file="data.json")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

#documents = SimpleDirectoryReader("data").load_data()
#index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Save & load index
index.storage_context.persist(persist_dir="./storage")
# Later:
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

# Query repeatedly
query_engine = index.as_query_engine()
print(query_engine.query("What is photosynthesis?"))