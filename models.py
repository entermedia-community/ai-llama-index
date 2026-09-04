from typing import List, Optional

from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage


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


class QueryDocsRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The query string.")
    parent_ids: List[str] = Field(..., min_length=1, description="List of parent document IDs to filter by.")


class FindDocIdsRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The query string.")


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=5, description="The prompt.")
    query: str = Field(..., min_length=5, description="The query string.")
    parent_ids: List[str] = Field(..., min_length=1, description="List of parent document IDs to filter by.")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The query string.")
    parent_ids: List[str] = Field(..., min_length=1, description="List of parent document IDs to filter by.")
    # [{role: "user", content: "What is the capital of France?"}, {role: "assistant", content: "The capital of France is Paris."}]
    chat_history: List[ChatMessage] = Field(..., description="List of chat messages representing the conversation history.")


class DeleteDocsRequest(BaseModel):
    node_ids: List[str] = Field(..., description="List of node IDs to delete.")
