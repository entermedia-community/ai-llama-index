import asyncio
import importlib
import pkgutil

from fastapi import FastAPI, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

import routers

app = FastAPI()
app.add_middleware(
    GZipMiddleware,
    minimum_size=500,
)

for module_info in sorted(pkgutil.iter_modules(routers.__path__), key=lambda module: module.name):
    if module_info.name.startswith("_"):
        continue

    module = importlib.import_module(f"{routers.__name__}.{module_info.name}")
    router = getattr(module, "router", None)
    if router is not None:
        app.include_router(router)


@app.exception_handler(asyncio.TimeoutError)
async def timeout_exception_handler(_, __):
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"error": "The operation timed out while waiting for model/vector-store response."},
    )