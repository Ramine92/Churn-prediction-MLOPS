from pydantic import BaseModel,Field
from typing import Optional

class PredictionRequest(BaseModel):
    customerID: Optional[str] = None
    gender : str
    SeniorCitizen: int = Field(ge=0,le=1)
    Partner : str
    Dependents : str
    tenure : int
    PhoneService : str
    MultipleLines : str
    InternetService : str
    OnlineSecurity : str
    OnlineBackup : str
    DeviceProtection: str
    TechSupport : str
    StreamingTV : str
    StreamingMovies : str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

class PredictionResponse(BaseModel):
    prediction : int
    probability : Optional[float] = None
    will_churn : Optional[bool] = None



