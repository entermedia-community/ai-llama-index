import asyncio

from fastapi import FastAPI, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from routers import root, save, chat, query, create_outline, create_structure, delete_document

app = FastAPI()
app.add_middleware(
    GZipMiddleware,
    minimum_size=500,
)

app.include_router(root.router)
app.include_router(save.router)
app.include_router(chat.router)
app.include_router(query.router)
app.include_router(create_outline.router)
app.include_router(create_structure.router)
app.include_router(delete_document.router)


@app.exception_handler(asyncio.TimeoutError)
async def timeout_exception_handler(_, __):
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"error": "The operation timed out while waiting for model/vector-store response."},
    )