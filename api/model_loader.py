import joblib
import numpy as np
from pathlib import Path
import os

# model path
BASE_DIR = Path(os.path.abspath(__file__)).parent.parent

MODEL_PATH = BASE_DIR / "src" / "models" / "models" / "Random_Forest.pkl"
SCALER_PATH = BASE_DIR / "src" / "models" / "models" / "scaler.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def get_risk_level(probability:float)->str:
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"
