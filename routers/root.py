from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Running AI Embedding Service"}
