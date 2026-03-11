# 🔄 Churn Prediction — End-to-End MLOps Pipeline

[![CI Pipeline](https://github.com/Ramine92/Churn-prediction-MLOPS/actions/workflows/ci.yml/badge.svg)](https://github.com/Ramine92/Churn-prediction-MLOPS/actions)

A complete MLOps project that predicts customer churn using Machine Learning, from exploratory analysis in a Jupyter notebook to a production-ready API deployed on the cloud.

**🔴 Live API** → [https://churn-prediction-api-5wk1.onrender.com](https://churn-prediction-api-5wk1.onrender.com)
**📊 MLflow Tracking** → [DagsHub Experiments](https://dagshub.com/Ramine92/Churn-prediction-MLOPS)

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [MLOps Pipeline](#mlops-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Model Training & Tracking](#model-training--tracking)
- [API Usage](#api-usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)

---

## Project Overview

This project addresses **customer churn prediction** for a telecom company using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset. The goal is to identify customers likely to leave, enabling proactive retention strategies.

### Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.75 | 0.52 | 0.8 | 0.635 | 0.85 |
| Random Forest (Tuned) | 0.77 | 0.55 | 0.72 | 0.63 | 0.84 |

> **Business Choice:** Random Forest was selected for its higher **Recall** (72%), meaning it catches more customers about to leave — which is more valuable than overall accuracy in a churn scenario.

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.11 |
| **ML Framework** | scikit-learn |
| **API** | FastAPI + Uvicorn |
| **Experiment Tracking** | MLflow + DagsHub |
| **Data & Model Versioning** | DVC + DagsHub S3 |
| **CI** | GitHub Actions |
| **CD** | Render (auto-deploy on push) |
| **Containerization** | Docker |
| **Testing** | pytest |

---

## MLOps Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
│  Notebook    │───▶│  train.py    │───▶│  FastAPI   │───▶│  Render  │
│  (EDA +      │    │  (Pipeline + │    │  (API +    │    │  (Live   │
│  Experiments)│    │  MLflow)     │    │  Tests)    │    │  Deploy) │
└─────────────┘    └──────────────┘    └───────────┘    └──────────┘
      ▲                   │                  │                │
      │                   ▼                  ▼                ▼
   Jupyter         DVC + DagsHub S3    GitHub Actions     Auto-deploy
                   (Model Versioning)  (CI: pytest)       (CD: on push)
```

### Pipeline Steps

1. **Data Exploration & Cleaning** — Jupyter notebook (`notebooks/Customer-Churn.ipynb`)
   - Handled missing values in `TotalCharges`
   - Dropped inconsistent rows (`tenure=0` with `TotalCharges=blank`)
   - Explored feature distributions and correlations

2. **Feature Engineering & Model Training** — `ml/models/train.py`
   - Built a scikit-learn `Pipeline` with `ColumnTransformer` (OneHotEncoder, StandardScaler, log transform)
   - Trained Logistic Regression and Random Forest models
   - Used `StratifiedKFold` (5-fold) cross-validation
   - Hyperparameter tuning via `RandomizedSearchCV`

3. **Experiment Tracking** — MLflow + DagsHub
   - All metrics (accuracy, precision, recall, F1, ROC-AUC) logged per run
   - Model parameters and artifacts tracked for reproducibility
   - Model comparison via DagsHub dashboard

4. **Model & Data Versioning** — DVC
   - Large files (`.pkl` models, `.csv` data) tracked by DVC
   - Stored on DagsHub S3 remote
   - `.dvc` pointer files committed to Git

5. **API Development** — FastAPI
   - `/predict/predict` — POST endpoint for churn prediction
   - `/health/health` — GET endpoint for health check + active model name
   - Dynamic model loading via environment variable (`MODEL_NAME`)

6. **Testing** — pytest
   - API health endpoint test
   - Run automatically on every push via GitHub Actions

7. **Containerization** — Docker
   - Multi-step Dockerfile optimized for cache (dependencies copied first)
   - DVC pull integrated into the build process

8. **CI/CD** — GitHub Actions + Render
   - **CI**: On every push to `main`, GitHub Actions runs tests
   - **CD**: Render auto-deploys the latest code from `main`

---

## Project Structure

```
ml-project/
├── Dockerfile                  # Docker image definition
├── .dockerignore               # Files excluded from Docker image
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI pipeline
├── .dvc/
│   └── config                  # DVC remote configuration
├── .env                        # Local environment variables (not in Git)
├── requirements.txt            # Python dependencies
├── app/                        # FastAPI application
│   ├── main.py                 # App entry point
│   ├── api/
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── routes/
│   │       ├── predict.py      # /predict endpoint
│   │       └── health.py       # /health endpoint
│   ├── core/
│   │   └── config.py           # Dynamic model path configuration
│   └── services/
│       └── predictor.py        # Model loading and prediction logic
├── ml/                         # Machine Learning code
│   ├── data/raw/               # Raw dataset (tracked by DVC)
│   ├── models/
│   │   ├── train.py            # Training script with MLflow tracking
│   │   └── artifacts/          # Saved .pkl models (tracked by DVC)
│   └── preprocessing.py        # Input preprocessing for API
├── notebooks/
│   └── Customer-Churn.ipynb    # Exploratory analysis notebook
└── tests/
    └── test_api.py             # API unit tests
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- DVC (`pip install dvc dvc-s3`)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ramine92/Churn-prediction-MLOPS.git
cd Churn-prediction-MLOPS

# 2. Create and activate a virtual environment
python -m venv ml_project_env
source ml_project_env/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull data and models from DVC remote
export AWS_ACCESS_KEY_ID=<your-dagshub-token>
export AWS_SECRET_ACCESS_KEY=<your-dagshub-token>
dvc pull

# 5. Create a .env file
echo "MODEL_NAME=Logistic_Regression" > .env

# 6. Run the API locally
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## Model Training & Tracking

### Train a model

```bash
# Edit MODEL_TO_TRAIN in ml/models/train.py ("Logistic_Regression" or "Random_Forest")
python ml/models/train.py
```

This will:
1. Load and clean the Telco Churn dataset
2. Train the selected model with a full preprocessing pipeline
3. Log parameters and metrics to **MLflow** (on DagsHub)
4. Save the model as a `.pkl` file locally

### Version a new model with DVC

```bash
dvc add ml/models/artifacts/Random_Forest_v1.pkl
dvc push
git add ml/models/artifacts/Random_Forest_v1.pkl.dvc
git commit -m "model: Update Random Forest"
git push
```

---

## API Usage

### Interactive Documentation

Visit `/docs` for the Swagger UI:
- **Local**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Production**: [https://churn-prediction-api-5wk1.onrender.com/docs](https://churn-prediction-api-5wk1.onrender.com/docs)

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check + active model name |
| POST | `/predict` | Churn prediction |

### Example: Predict Churn

```bash
curl -X POST https://churn-prediction-api-5wk1.onrender.com/predict/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 840.50
  }'
```

### Response

```json
{
  "prediction": 1,
  "probability": 78.5,
  "will_churn": true
}
```

---

## CI/CD Pipeline

### CI — GitHub Actions

On every push to `main`:
1. Checkout code
2. Set up Python 3.11
3. Install dependencies
4. Pull models via DVC
5. Run tests with pytest

### CD — Render (Auto-Deploy)

Render is connected to this GitHub repository. On every push to `main`:
1. Render pulls the latest code
2. Installs dependencies + downloads models from DagsHub S3
3. Starts the FastAPI server
4. API is live at the public URL

### Switching Models in Production

To switch the active model in production, update the `MODEL_NAME` environment variable in the Render dashboard:
- `Logistic_Regression` — Higher precision, fewer false alarms
- `Random_Forest` — Higher recall, catches more churners

No code change or redeployment needed — just restart the service.

---

## Environment Variables

| Variable | Where | Purpose |
|---|---|---|
| `MODEL_NAME` | `.env` (local) / Render dashboard (prod) | Which model to load |
| `AWS_ACCESS_KEY_ID` | `.dvc/config.local` / Render / GitHub Secrets | DagsHub S3 authentication |
| `AWS_SECRET_ACCESS_KEY` | `.dvc/config.local` / Render / GitHub Secrets | DagsHub S3 authentication |

---

## Author

**Ramine** — [GitHub](https://github.com/Ramine92) · [DagsHub](https://dagshub.com/Ramine92)
