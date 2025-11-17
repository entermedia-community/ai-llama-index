from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from document_maker import DocumentMaker
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

# Global variables
vector_store = None
index = None

Settings.llm = OpenAILike(
    api_base="http://142.113.71.170:36238/v1",
    is_chat_model=True,
    is_function_calling_model=True
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

client = QdrantClient(path="./qdrant_db")


@asynccontextmanager
async def lifespan(app: FastAPI):   
    global vector_store, index

    vector_store = QdrantVectorStore(client=client, collection_name="documents")

    try:
        index = VectorStoreIndex.from_vector_store(vector_store)
        print("Loaded existing index")
    except:
        print("Creating new index with storage_context")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context 
        )
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Running AI Embedding Service"}

class CreateEmbeddingRequest(BaseModel):
    id: str
    data: str
    metadata: dict | None = None

@app.post("/save")
async def embed_document(data: CreateEmbeddingRequest):
    id = data.id
    if not id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Document ID is required."}
        )
    text = data.data
    if not text or text.strip() == "":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Data text is required."}
        )
    metadata = data.metadata or {}

    doc_maker = DocumentMaker(
        id=id,
        page_label=metadata.get("page_label"),
        file_name=metadata.get("file_name"),
        file_type=metadata.get("file_type"),
        creation_date=metadata.get("creation_date"),
    )
    try:
        document = doc_maker.create_document(text)
        index.insert(document)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"message": f"Document with ID {id} embedded successfully."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )


class QueryDocsRequest(BaseModel):
    query: str
    doc_ids: list[str]

@app.post("/query")
async def query_docs(data: QueryDocsRequest):
    try:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="id", operator=FilterOperator.IN, value=data.doc_ids)
            ]
        )

        query_engine = index.as_query_engine(filters=filters)
        
        response = query_engine.query(data.query)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
            "query": data.query,
            "answer": str(response),
            "sources": [
                {
                    **node.node.metadata,
                    "score": node.score,
                }
                for node in response.source_nodes
            ]
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )
