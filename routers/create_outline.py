import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from llama_index.core import PromptTemplate, VectorStoreIndex
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

from core import (
    get_collection_name,
    get_vector_store,
    heavy_request_semaphore,
    llm,
    run_blocking,
    INDEX_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from models import QueryDocsRequest

logger = logging.getLogger(__name__)

router = APIRouter()

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


@router.post("/create_outline")
async def create_outline(
    data: QueryDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    async with heavy_request_semaphore:

        vector_store = await run_blocking(get_vector_store, x_customerkey, timeout=INDEX_TIMEOUT_SECONDS)
        index = VectorStoreIndex.from_vector_store(vector_store, use_async=True)

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
                use_async=True,
                timeout=INDEX_TIMEOUT_SECONDS,
            )

            nodes = await asyncio.wait_for(retriever.aretrieve(data.query), timeout=REQUEST_TIMEOUT_SECONDS)
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
