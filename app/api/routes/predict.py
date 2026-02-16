from fastapi import APIRouter,HTTPException,status
from app.services.predictor import predict as predict_service
from app.api.schemas import PredictionResponse,PredictionRequest
router = APIRouter()

@router.post("/predict",response_model=PredictionResponse)
async def predict_endpoint(data:PredictionRequest):
    try:
        customer_data = data.dict()
        prediction,probability = predict_service(customer_data)
        if hasattr(prediction,"item"):
            prediction = prediction.item()

        response = PredictionResponse(prediction=int(prediction),probability=probability*100,will_churn=(prediction==1))
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail=f"prediction failed: {str(e)}")


