import pandas as pd
import numpy as np 

def process_input(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop("customerID",axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
    drop_indexes = df[(df["TotalCharges"] == 0) & (df["tenure"] > 0)].index
    df = df.drop(drop_indexes)
    return df