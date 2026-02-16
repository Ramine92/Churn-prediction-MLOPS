from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_VERSION = "v1"
MODEL_PATH = BASE_DIR / "ml" / "models" / "artifacts"/f"model_{MODEL_VERSION}.pkl"

if not MODEL_PATH.exists():
    print(f"Model {MODEL_VERSION} not found at {MODEL_PATH}")
    alt_path = Path("/Users/graiaraminmohamedryad/Desktop/1-END-TO-END/ml-project/ml/models/artifacts/model_v1.pkl")
    if alt_path.exists():
        MODEL_PATH = alt_path