from typing import Optional, List
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import List, Optional
from fastapi import FastAPI, status, Header, Depends
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from utils.document_maker import DocumentMaker

from db_manager import IndexRegistry

app = FastAPI()

registry = IndexRegistry(dim=1024)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.llm = OpenAILike(
    api_base="http://localhost:7600/",
    is_chat_model=True,
    is_function_calling_model=True
)

Settings.embed_model = HuggingFaceEmbedding(
  model_name="BAAI/bge-m3"
)

def get_collection_name(x_customerkey: Optional[str] = Header(None)):
    if not x_customerkey.isalnum():
        raise ValueError("Invalid customer key.")
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
    
    index = registry.get(collection_name=x_customerkey)

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
    node_ids = {}
    print("Adding pages for document ID:", doc_id)
    for data in all_data.pages:
        page_id = data.page_id
        if not page_id:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Document ID is required."}
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
            index.insert(document)
            processed.add(page_id)
            node_id = getattr(document, "id_", None)
            if node_id is None and hasattr(document, "get_doc_id"):
                node_id = document.get_doc_id()
            if node_id is None and hasattr(document, "doc_id"):
                node_id = document.doc_id
            if node_id is not None:
                node_ids[page_id] = node_id
            print("Added page ID:", page_id)
        except Exception as e:
            failed.add(page_id)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"Document {doc_id} embedded successfully.",
            "processed": list(processed),
            "skipped": list(skipped),
            "failed": list(failed),
            "node_ids": node_ids,
        }
    )
    
class QueryDocsRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The query string.")
    parent_ids: List[str] = Field(..., min_length=1, description="List of parent document IDs to filter by.")

@app.post("/query")
async def query_docs(
    data: QueryDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    
    index = registry.get(collection_name=x_customerkey)

    try:
        if(len(data.parent_ids) == 1):
            operator = FilterOperator.EQ
            value = data.parent_ids[0]
        else:
            operator = FilterOperator.IN
            value = data.parent_ids
        
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="parent_id", operator=operator, value=value)
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
        print("Error during query:", str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )
    
# class DeleteDocsRequest(BaseModel):
#     page_ids: str = Field(..., min_length=1, description="The page IDs to delete.")

# @app.post("/delete-pages")
# async def delete_by_doc_id(
#     doc: DeleteDocsRequest,
#     x_customerkey: Optional[str] = Depends(get_collection_name)
# ):
#     index = registry.get(collection_name=x_customerkey)
#     try:
#         index.delete_ref_doc(doc.page_ids, delete_from_docstore=True)
#         return JSONResponse(
#             status_code=status.HTTP_200_OK,
#             content={"message": f"Documents with page_ids {doc.page_ids} deleted successfully."}
#         )
#     except Exception as e:
#         print("Error during deletion:", str(e))
#         return JSONResponse(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             content={"error": str(e)}
#         )