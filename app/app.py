from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd

# -------------------------------------------------------
# Initialize FastAPI
# -------------------------------------------------------
app = FastAPI(
    title="Dementia Classification API",
    description="Predict dementia classification using an XGBoost model.",
    version="1.0.0"
)

# -------------------------------------------------------
# Load model
# -------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
model = joblib.load(MODEL_PATH)

# If model supports predict_proba, store flag
HAS_PROBA = hasattr(model, "predict_proba")

# -------------------------------------------------------
# Input schema (Pydantic)
# -------------------------------------------------------
class Input(BaseModel):
    Visit: float
    MR_Delay: float
    M_F: float
    Age: float
    EDUC: float
    SES: float
    MMSE: float
    CDR: float
    eTIV: float
    nWBV: float
    ASF: float
    ABV: float
    CII: float
    CDR_RATE: float

# -------------------------------------------------------
# Health check
# -------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------
@app.post("/predict")
def predict(input: Input):
    """
    Returns:
        prediction (0/1)
        probability (if available)
    """

    # Map Pydantic fields → model's original column names
    data = {
        "Visit": input.Visit,
        "MR Delay": input.MR_Delay,
        "M/F": input.M_F,
        "Age": input.Age,
        "EDUC": input.EDUC,
        "SES": input.SES,
        "MMSE": input.MMSE,
        "CDR": input.CDR,
        "eTIV": input.eTIV,
        "nWBV": input.nWBV,
        "ASF": input.ASF,
        "ABV": input.ABV,
        "CII": input.CII,
        "CDR_RATE": input.CDR_RATE
    }

    # Ensure correct feature order
    df = pd.DataFrame([data], columns=model.get_booster().feature_names)

    # Prediction
    y_pred = int(model.predict(df)[0])

    response = {"prediction": y_pred}

    # Add probability if available
    if HAS_PROBA:
        proba = model.predict_proba(df)[0][1]   # Probability of class 1
        response["probability"] = float(proba)

    return response
