from fastapi import APIRouter
from app.core.config import MODEL_NAME
router = APIRouter()

@router.get("/")
def health():
    return {"status":"ok","model_name":MODEL_NAME}