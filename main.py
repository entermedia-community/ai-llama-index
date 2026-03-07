import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
from functools import partial, lru_cache

from typing import Optional, List
import logging
from threading import Lock

from fastapi import FastAPI, status, Header, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from qdrant_client.http.models import Filter, FieldCondition, MatchAny


from llama_index.vector_stores.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from utils.document_maker import DocumentMaker

from fastapi.middleware.gzip import GZipMiddleware


llm = OpenAILike(
    api_base="http://0.0.0.0:7600/", # Server uses local LLM
    # api_base="https://llamat.emediaworkspace.com/", # Use this for testing locally with the remote LLM
    is_chat_model=True,
    is_function_calling_model=True
)

class Outlines(BaseModel):
    outline: List[str] = Field(..., description="List of outline sections extracted from the document.")

s_llm = llm.as_structured_llm(output_cls=Outlines)

Settings.llm = llm

Settings.embed_model = HuggingFaceEmbedding(
  model_name="BAAI/bge-m3"
)

client = QdrantClient(
    host="localhost",
    port=6333,
    # host="74.48.140.178", 
    # port=27054
)

VECTOR_SIZE = 1024

def ensure_collection(name: str):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

@lru_cache(maxsize=32)
def get_vector_store(collection: str) -> QdrantVectorStore:
    ensure_collection(collection)
    return QdrantVectorStore(client=client, collection_name=collection)



app = FastAPI()
app.add_middleware(
    GZipMiddleware,
    minimum_size=500, 
)

logger = logging.getLogger(__name__)

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

@app.get("/")
async def root():
    return {"message": "Running AI Embedding Service"}

class CreateEmbeddingData(BaseModel):
    page_id: str = Field(..., min_length=1, description="The page ID.")
    text: str = Field(..., min_length=1, description="The text content of the page.")
    page_label: str | None = Field(None, description="The label of the page.")

class CreateEmbeddingRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, description="The document ID.")
    file_name: str | None = Field(None, description="The file name.")
    file_type: str | None = Field(None, description="The file type.")
    creation_date: str | None = Field(None, description="The creation date.")
    pages: List[CreateEmbeddingData] = Field(..., min_length=1, description="List of pages to embed.")

@app.post("/save")
async def embed_document(
    all_data: CreateEmbeddingRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    async with heavy_request_semaphore:
        
        vector_store = await run_blocking(get_vector_store, get_collection_name(x_customerkey), timeout=INDEX_TIMEOUT_SECONDS)
        index = VectorStoreIndex.from_vector_store(vector_store)

        doc_id = all_data.doc_id
        file_name = all_data.file_name
        file_type = all_data.file_type
        creation_date = all_data.creation_date

        if not doc_id:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Document ID is required."}
            )

        processed = set()
        failed = set()
        skipped = set()
        logger.info("Adding pages for document ID: %s", doc_id)
        for data in all_data.pages:
            page_id = data.page_id
            if not page_id:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "Document ID is required."}
                )

            await run_blocking(
                index.delete,
                doc_id=page_id,
                delete_from_docstore=True,
            )
            page_label = data.page_label
            text = data.text

            if not text or text.strip() == "":
                skipped.add(page_id)
                continue

            doc_maker = DocumentMaker(
                id=page_id,
                parent_id=doc_id,
                page_label=page_label,
                file_name=file_name,
                file_type=file_type,
                creation_date=creation_date,
            )
            try:
                document = doc_maker.create_document(text)
                await run_blocking(index.insert, document)
                processed.add(page_id)

                logger.info("Added page ID: %s", page_id)
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                failed.add(page_id)
                logger.error("Error embedding page %s of document %s: %s", page_id, doc_id, str(e))

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Document {doc_id} embedded successfully.",
                "processed": list(processed),
                "skipped": list(skipped),
                "failed": list(failed),
            }
        )

    
@app.exception_handler(asyncio.TimeoutError)
async def timeout_exception_handler(_, __):
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"error": "The operation timed out while waiting for model/vector-store response."},
    )
    
class QueryDocsRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The query string.")
    parent_ids: List[str] = Field(..., min_length=1, description="List of parent document IDs to filter by.")

@app.post("/query")
async def query_docs(
    data: QueryDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    async with heavy_request_semaphore:
        
        vector_store = await run_blocking(get_vector_store, get_collection_name(x_customerkey), timeout=INDEX_TIMEOUT_SECONDS)
        index = VectorStoreIndex.from_vector_store(vector_store)

        try:
            filters = Filter(
                must=[
                    FieldCondition(
                        key="parent_id",
                        match=MatchAny(
                            any=data.parent_ids
                        )
                    )
                ]
            )

            query_engine = await run_blocking(
                index.as_query_engine,
                vector_store_kwargs={"qdrant_filters": filters},
                timeout=INDEX_TIMEOUT_SECONDS,
            )
            response = await run_blocking(query_engine.query, data.query)

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
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error("Error during query_docs: %s", str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": str(e)}
            )

SECTION_HEADERS_PROMPT = PromptTemplate(
    """You are a document structure expert. Given a context and a user query, 
generate a list of relevant section headers that would best organize the information 
needed to answer the query.

<context>
{context}
</context>

<user_query>
{query}
</user_query>

Instructions:
- Generate section headers that are directly relevant to the query
- Headers should logically organize the content from the context
- Be concise and descriptive (3-7 words per header)
- Order headers in a logical reading flow
- Return ONLY a JSON array of strings, no explanation

Example output format:
["Introduction to Topic", "Key Concepts", "How It Works", "Common Use Cases", "Summary"]

Output:"""
)

@app.post("/create_outline")
async def create_outline(
    data: QueryDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    async with heavy_request_semaphore:
        
        vector_store = await run_blocking(get_vector_store, get_collection_name(x_customerkey), timeout=INDEX_TIMEOUT_SECONDS)
        index = VectorStoreIndex.from_vector_store(vector_store)

        try:
            filters = Filter(
                must=[
                    FieldCondition(
                        key="parent_id",
                        match=MatchAny(
                            any=data.parent_ids
                        )
                    )
                ]
            )

            retriever = await run_blocking(
                index.as_retriever,
                vector_store_kwargs={"qdrant_filters": filters},
                timeout=INDEX_TIMEOUT_SECONDS,
            )

            nodes = await run_blocking(retriever.retrieve, data.query)
            context = ""
            for node in nodes:
                context += f"Page ID: {node.node.metadata.get('id', 'N/A')}, Page Label: {node.node.metadata.get('page_label', 'N/A')}\n"
                context += f"Content: {node.node.get_content()}\n\n"

            response = await run_blocking(
                llm.predict,
                SECTION_HEADERS_PROMPT,
                context=context,
                query=data.query,
            )

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"outline": json.loads(response)}
            )
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error("Error during create_outline: %s", str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": str(e)}
            )
    
class DeleteDocsRequest(BaseModel):
    node_ids: List[str] = Field(..., description="List of node IDs to delete.")
    
@app.post("/delete_document")
async def delete_document(
    data: DeleteDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    async with heavy_request_semaphore:
        
        vector_store = await run_blocking(get_vector_store, get_collection_name(x_customerkey), timeout=INDEX_TIMEOUT_SECONDS)
        index = VectorStoreIndex.from_vector_store(vector_store)

        for node_id in data.node_ids:
            await run_blocking(index.delete, doc_id=node_id, delete_from_docstore=True)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Nodes deleted successfully."}
    )