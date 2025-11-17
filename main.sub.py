import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import nest_asyncio
# nest_asyncio.apply()

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from document_maker import DocumentMaker

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager

Settings.llm = OpenAILike(
    api_base="http://142.113.71.170:36238/v1",
    is_chat_model=True,
    is_function_calling_model=True
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

client = QdrantClient(path="./qdrant_db")

query_engine_tools = []

@asynccontextmanager
async def startup_event(app: FastAPI):
    global vector_store, index, query_engine_tools
    
    vector_store = QdrantVectorStore(client=client, collection_name="documents")

    try:
        index = VectorStoreIndex.from_vector_store(vector_store, use_async=True)
        print("Loaded existing index")
    except:
        print("Creating new index with storage_context")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context ,
            use_async=True
        )
    
    vector_query_engine = index.as_query_engine(
        similarity_top_k=10,
        response_mode="tree_summarize", #slow but better higher accuracy (other options("refine", "compact", "simple_summarize"))
    )
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="document_search",
                description="Useful for searching through documents to find relevant information.",
            ),
        ),
    ]
    docs_tool = QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="documents",
            description=(
                "Search across all documents in the database. "
                "Automatically finds the most relevant documents for any query."
            ),
        ),
    )
    yield
    

app = FastAPI(lifespan=startup_event)

@app.get("/")
async def root():
    return {"message": "Running AI Embedding Service"}

class CreateEmbeddingRequest(BaseModel):
    id: str
    data: str
    metadata: dict | None = None

@app.post("/save/")
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

def search_documents(data: QueryDocsRequest) -> str:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="id", operator=FilterOperator.IN, value=data.doc_ids)
        ]
    )

    retriever = index.as_retriever(filters=filters)

    response = retriever.retrieve(data.query)
    return response

@app.post("/search/")
async def search_docs(data: QueryDocsRequest):
    return str(search_documents(data))

@app.post("/query/")
async def query_docs(data: QueryDocsRequest):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="id", operator=FilterOperator.IN, value=data.doc_ids)
        ]
    )

    query_engine = SubQuestionQueryEngine.from_defaults(
        question_gen=Settings.llm,
        query_engine_tools=query_engine_tools,
        # # filters=filters,
        # use_async=True,
    )
    response = query_engine.query(data.query)
    return str(response)

    # def multiply(a: float, b: float) -> float:
    #     """Useful for multiplying two numbers."""
    #     return a * b

    # agent = FunctionAgent(
    #     tools=[multiply, search_documents],
    #     system_prompt="""You are a helpful assistant that can perform calculations
    #     and search through documents to answer questions.""",
    # )

    # response = await agent.run(data.query, data)
    # return str(response)

