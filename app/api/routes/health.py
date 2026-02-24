from fastapi import APIRouter
from app.core.confing import MODEL_NAME
router = APIRouter()

@router.get("/health")
def health():
    return {"status":"ok","model_name":MODEL_NAME}