from pathlib import Path
import os
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR/".env")
MODEL_NAME = os.getenv("MODEL_NAME","Random_Forest")
MODEL_VERSION = "v1"
MODEL_PATH = BASE_DIR / "ml" / "models" / "artifacts"/f"{MODEL_NAME}_{MODEL_VERSION}.pkl"

if not MODEL_PATH.exists():
    print(f"Model {MODEL_VERSION} not found at {MODEL_PATH}")