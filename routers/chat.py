import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

from core import (
    get_collection_name,
    get_vector_store,
    heavy_request_semaphore,
    run_blocking,
    INDEX_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from models import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat")
async def chat_docs(
    data: ChatRequest,
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

            memory = ChatMemoryBuffer.from_defaults(chat_history=data.chat_history)

            chat_engine = await run_blocking(
                index.as_chat_engine,
                chat_mode="context",
                memory=memory,
                system_prompt="""You are a learning assistant.
The conversation contains authoritative learning material,
multiple-choice questions, learner answers, and explanations.
Answer the learner's follow-up questions based primarily on
the learning material provided in the conversation.""",
                vector_store_kwargs={"qdrant_filters": filters},
                use_async=True,
                timeout=INDEX_TIMEOUT_SECONDS,
            )
            response = await asyncio.wait_for(chat_engine.achat(data.query), timeout=REQUEST_TIMEOUT_SECONDS)

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
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error("Error during chat_docs: %s", str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": str(e)}
            )
