import joblib
from pathlib import Path
import os

# model path
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "src" / "models" / "models" / "Random_Forest_v2.pkl"
SCALER_PATH = BASE_DIR / "src" / "models" / "models" / "scaler_v2.pkl"
ENCODERS_PATH = BASE_DIR / "src" / "models" / "models" / "encoders_v2.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
    print(f"Loaded scaler from: {SCALER_PATH}")
    print(f"Loaded encoders from: {ENCODERS_PATH}")
    return model, scaler, encoders


def get_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"
