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
)
from models import CreateEmbeddingRequest
from utils.document_maker import DocumentMaker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/save")
async def embed_document(
    all_data: CreateEmbeddingRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    async with heavy_request_semaphore:

        vector_store = await run_blocking(get_vector_store, x_customerkey, timeout=INDEX_TIMEOUT_SECONDS)
        index = VectorStoreIndex.from_vector_store(vector_store, use_async=True)

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
