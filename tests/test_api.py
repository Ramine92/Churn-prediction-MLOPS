from fastapi.testclient import TestClient
from app.core.config import MODEL_NAME
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health/health")
    assert response.status_code == 200
    assert response.json() == {"status":"ok","model_name":MODEL_NAME}


