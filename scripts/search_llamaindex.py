from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# Load documents and create index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Save index
index.storage_context.persist(persist_dir="./storage")

# Load index from storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

# Query the index
query_engine = index.as_query_engine()
print(query_engine.query("What is photosynthesis?"))
