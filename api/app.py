"""FastAPI application for Titanic survival prediction."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas import PassengerInput, PredictionOutput, HealthResponse

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict passenger survival on the Titanic using ML.",
    version="1.0.0",
)

# Load model and pipeline at startup
MODEL = None
PIPELINE = None


@app.on_event("startup")
def load_artifacts():
    global MODEL, PIPELINE
    model_path = ROOT / "models" / "best_model.joblib"
    pipeline_path = ROOT / "models" / "pipeline.joblib"
    if model_path.exists() and pipeline_path.exists():
        MODEL = joblib.load(model_path)
        PIPELINE = joblib.load(pipeline_path)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", model_loaded=MODEL is not None)


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    if MODEL is None or PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build title for Name field
    if passenger.sex == "female":
        title = "Mrs." if passenger.age >= 18 else "Miss."
    else:
        title = "Master." if passenger.age < 13 else "Mr."

    input_df = pd.DataFrame([{
        "PassengerId": 0,
        "Pclass": passenger.pclass,
        "Name": f"Passenger, {title} API",
        "Sex": passenger.sex,
        "Age": passenger.age,
        "SibSp": passenger.sibsp,
        "Parch": passenger.parch,
        "Ticket": "API",
        "Fare": passenger.fare,
        "Cabin": None,
        "Embarked": passenger.embarked,
    }])

    features = PIPELINE.named_steps["features"]
    scaler = PIPELINE.named_steps["scaler"]
    X = scaler.transform(features.transform(input_df))

    prob = float(MODEL.predict_proba(X)[0][1])
    survived = prob >= 0.5

    factors = []
    if passenger.sex == "female":
        factors.append("Female passengers had 74.2% survival rate")
    else:
        factors.append("Male passengers had only 18.9% survival rate")
    factors.append(f"Class {passenger.pclass}: {[0, 63.0, 47.3, 24.2][passenger.pclass]}% survival")

    family = passenger.sibsp + passenger.parch + 1
    if family == 1:
        factors.append("Solo travelers had 30.4% survival")
    elif family <= 4:
        factors.append("Small families had 57.9% survival")

    return PredictionOutput(survived=survived, probability=round(prob, 4), factors=factors)
