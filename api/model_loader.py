import joblib
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

REPO_ID = "shivamRamanuj/loan-default-model"

# model path
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "src" / "models" / "models"

MODEL_PATH = MODEL_DIR / "Random_Forest_v2.pkl"
SCALER_PATH = MODEL_DIR / "scaler_v2.pkl"
ENCODERS_PATH = MODEL_DIR / "encoders_v2.pkl"


def download_if_missing(filename, local_path):
    if not local_path.exists():
        print(f"Downloading {filename} from Hugging Face...")
        os.makedirs(local_path.parent, exist_ok=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(local_path.parent)
        )
        print(f"Downloaded {filename} successfully.")
    else:
        print(f"Found {filename} locally, skipping download.")

def load_model():
    download_if_missing("Random_Forest_v2.pkl", MODEL_PATH)
    download_if_missing("scaler_v2.pkl", SCALER_PATH)
    download_if_missing("encoders_v2.pkl", ENCODERS_PATH)
    
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
