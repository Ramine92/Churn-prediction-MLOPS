from fastapi import FastAPI,APIRouter
from app.api.routes import predict,health
app = FastAPI(title="Ml Churn Prediction Project")

@app.get("/")
def root():
    return {"message":"Welcome to Ml Churn Prediction Project"}
app.include_router(predict.router,prefix="/predict")
app.include_router(health.router,prefix="/health")
