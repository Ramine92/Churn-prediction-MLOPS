import joblib
import pandas as pd
from app.core.config import MODEL_PATH
from ml.preprocessing import process_input
try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully: {type(pipeline)}")
except Exception as e:
    print(f"Model could not be loaded")
    pipeline = None

def predict(customer_data: dict):
    if pipeline is None:
        raise RuntimeError(f"Model not loaded cannot make predictions")
    try:
        df = pd.DataFrame([customer_data])
        df = process_input(df)
        prediction = pipeline.predict(df)
        probability = pipeline.predict_proba(df)[:,1][0]
        return prediction,probability
    except Exception as e:
        raise BaseException(f'Prediction Error {e}')
