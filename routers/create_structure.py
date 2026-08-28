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
from models import PromptRequest

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/create_structure")
async def create_structure(
    data: QueryDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    async with heavy_request_semaphore:
        promopt = PromptTemplate(data.prompt)
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
                prompt,
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
            logger.error("Error during create_structure: %s", str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": str(e)}
            )
