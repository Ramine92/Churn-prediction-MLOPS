import pandas as pd
import numpy as np 
from pathlib import Path
import joblib
import dagshub
import mlflow
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

dagshub.init(repo_owner="Ramine92",repo_name="Churn-prediction-MLOPS",mlflow=True)
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data"/ "raw"/ "Telco-Customer-Churn.csv"
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def clean_data(df):
    df = df.drop("customerID",axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
    drop_indexes = df[(df["TotalCharges"] == 0) & (df["tenure"] > 0)].index
    df = df.drop(drop_indexes)
    if "Churn" in df.columns:
        y = df["Churn"].map({"Yes":1,"No":0})
        X = df.drop("Churn",axis=1)
        return X,y
    else:
        return df,None

def get_pipeline(X):
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    normal_features = ["tenure","MonthlyCharges"]
    skewed_features = ["TotalCharges"]

    cat_transformer = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="constant",fill_value="missing")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    normal_transformer = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="mean")),
        ("scaler",StandardScaler())
    ])

    skewed_transformer = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("log",FunctionTransformer(np.log1p))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",cat_transformer,cat_features),
            ("normal",normal_transformer,normal_features),
            ("skewed",skewed_transformer,skewed_features)
        ]
    )
    pipeline = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(class_weight="balanced",random_state=42))
    ])
    return pipeline

def train_model(X,y):
    #split data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    #get pipeline
    pipeline = get_pipeline(X_train)
    #fit model
    pipeline.fit(X_train,y_train)
    #stratified k-fold cross_val
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    #metrics
    scoring = ["accuracy","precision","recall","f1","roc_auc"]
    cv_results = cross_validate(pipeline,X_train,y_train,cv=skf,scoring=scoring,return_train_score=True)
    # === Logging MLflow ===
    mlflow.log_metric("recall", cv_results['test_recall'].mean())
    mlflow.log_metric("precision", cv_results['test_precision'].mean())
    mlflow.log_metric("f1", cv_results['test_f1'].mean())
    mlflow.log_metric("roc_auc", cv_results['test_roc_auc'].mean())
    mlflow.log_metric("accuracy", cv_results['test_accuracy'].mean())
    return pipeline
    
def save_model(model):
    artifacts_path = BASE_DIR / "models" / "artifacts"
    artifacts_path.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,artifacts_path / "model_v1.pkl")
    print("model saved successfully")

if __name__ == "__main__":
    df = load_data()
    X,y = clean_data(df)
    mlflow.set_experiment("Baseline_Models")

    with mlflow.start_run():
        print("Démarrage de la Run MLflow...")

    mlflow.log_param("model_type","Logistic Regression")

    mlflow.log_param("class_weight","balanced")
    model = train_model(X,y)
    
    mlflow.sklearn.log_model(model,"logistic_regression_model")
    save_model(model)
