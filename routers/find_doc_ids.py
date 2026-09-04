import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from llama_index.core import VectorStoreIndex

from core import (
    get_collection_name,
    get_vector_store,
    heavy_request_semaphore,
    run_blocking,
    INDEX_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/findDocIds")
async def find_doc_ids(
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    async with heavy_request_semaphore:
        vector_store = await run_blocking(get_vector_store, x_customerkey, timeout=INDEX_TIMEOUT_SECONDS)

        try:
            doc_ids = set()
            offset = None

            while True:
                records, offset = await asyncio.wait_for(
                    vector_store.aclient.scroll(
                        collection_name=x_customerkey,
                        limit=256,
                        offset=offset,
                        with_payload=["parent_id"],
                        with_vectors=False,
                    ),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                doc_ids.update(
                    record.payload["parent_id"]
                    for record in records
                    if record.payload and record.payload.get("parent_id")
                )

                if offset is None:
                    break

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"doc_ids": sorted(doc_ids)}
            )
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error("Error during find_doc_ids: %s", str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": str(e)}
            )