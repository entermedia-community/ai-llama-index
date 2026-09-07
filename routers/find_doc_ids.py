import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from llama_index.core import VectorStoreIndex
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

from core import (
    get_collection_name,
    get_vector_store,
    heavy_request_semaphore,
    run_blocking,
    INDEX_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from models import QueryDocsRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/findDocIds")
async def find_doc_ids(
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
                        match=MatchAny(any=data.parent_ids),
                    )
                ]
            )
            allowed_parent_ids = set(data.parent_ids)

            similarity_top_k = max(len(data.parent_ids) * 2, 10)

            retriever = await run_blocking(
                index.as_retriever,
                similarity_top_k=similarity_top_k,
                retriever_mode="embedding",
                vector_store_kwargs={"qdrant_filters": filters},
                use_async=True,
                timeout=INDEX_TIMEOUT_SECONDS,
            )
            nodes = await asyncio.wait_for(
                retriever.aretrieve(data.query),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            matches = [
                {
                    "metadata": node.node.metadata,
                    "score": node.score,
                }
                for node in nodes
                if node.node.metadata.get("parent_id") in allowed_parent_ids
                and (
                    data.score_threshold is None
                    or node.score is None
                    or node.score >= data.score_threshold
                )
            ]
            doc_ids = [match["metadata"] for match in matches]

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"doc_ids": doc_ids, "matches": matches}
            )
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error("Error during find_doc_ids: %s", str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": str(e)}
            )