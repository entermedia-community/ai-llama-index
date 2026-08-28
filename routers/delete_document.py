from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from llama_index.core import VectorStoreIndex

from core import get_collection_name, get_vector_store, heavy_request_semaphore, run_blocking, INDEX_TIMEOUT_SECONDS
from models import DeleteDocsRequest

router = APIRouter()


@router.post("/delete_document")
async def delete_document(
    data: DeleteDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    async with heavy_request_semaphore:

        vector_store = await run_blocking(get_vector_store, x_customerkey, timeout=INDEX_TIMEOUT_SECONDS)
        index = VectorStoreIndex.from_vector_store(vector_store)

        for node_id in data.node_ids:
            await run_blocking(index.delete, doc_id=node_id, delete_from_docstore=True)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Nodes deleted successfully."}
    )
