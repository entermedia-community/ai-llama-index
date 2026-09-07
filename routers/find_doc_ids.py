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
            allowed_parent_ids = set(data.parent_ids)

            filters = Filter(
                must=[
                    FieldCondition(
                        key="parent_id",
                        match=MatchAny(any=allowed_parent_ids),
                    )
                ]
            )
            
            similarity_top_k = max(len(allowed_parent_ids) * 2, 10)

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

            threshold = 0.5 if data.score_threshold is None else data.score_threshold

            matched_parents = [
                node.node.metadata["parent_id"]
                for node in nodes
                if node.score is not None and node.score >= threshold
            ]

            matched_parents = list(dict.fromkeys(matched_parents))

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"parent_ids": matched_parents}
            )
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error("Error during find_doc_ids: %s", str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": str(e)}
            )