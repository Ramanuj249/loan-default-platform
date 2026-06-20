from fastapi import FastAPI
import pandas as pd

try:
    from api.schemas import LoanPredictionInput, LoanPredictionOutput
    from api.model_loader import load_model, get_risk_level
except ImportError:
    from schemas import LoanPredictionInput, LoanPredictionOutput
    from model_loader import load_model, get_risk_level

import numpy as np
from contextlib import asynccontextmanager

# Encoding mappings (from LabelEncoder during training)
GENDER_MAP = {"F": 0, "M": 1}
EDUCATION_MAP = {"Academic degree": 0, "Higher education": 1, "Incomplete higher": 2, "Lower secondary": 3, "Secondary / secondary special": 4}
FAMILY_STATUS_MAP = {"Civil marriage": 0, "Married": 1, "Separated": 2, "Single / not married": 3, "Widow": 5}
HOUSING_TYPE_MAP = {"Co-op apartment": 0, "House / apartment": 1, "Municipal apartment": 2, "Office apartment": 3, "Rented apartment": 4, "With parents": 5}
INCOME_TYPE_MAP = {"Businessman": 0, "Commercial associate": 1, "Pensioner": 3, "State servant": 4, "Working": 7}
CONTRACT_TYPE_MAP = {"Cash loans": 0, "Revolving loans": 1}
FLAG_MAP = {"Yes": 1, "No": 0}

# Global variables to store model and scaler
model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    model, scaler = load_model()
    print("Model loaded successfully!")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts probability of loan default",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=LoanPredictionOutput)
def predict(input_data: LoanPredictionInput):
    features = pd.DataFrame([[
        0,  # SK_ID_CURR
        CONTRACT_TYPE_MAP[input_data.NAME_CONTRACT_TYPE],  # NAME_CONTRACT_TYPE
        GENDER_MAP[input_data.CODE_GENDER],  # CODE_GENDER
        FLAG_MAP[input_data.FLAG_OWN_CAR],  # FLAG_OWN_CAR
        FLAG_MAP[input_data.FLAG_OWN_REALTY],  # FLAG_OWN_REALTY
        input_data.CNT_CHILDREN,  # CNT_CHILDREN
        input_data.AMT_INCOME_TOTAL,  # AMT_INCOME_TOTAL
        input_data.AMT_CREDIT,  # AMT_CREDIT
        input_data.AMT_ANNUITY,  # AMT_ANNUITY
        input_data.AMT_GOODS_PRICE,  # AMT_GOODS_PRICE
        0,  # NAME_TYPE_SUITE
        INCOME_TYPE_MAP[input_data.NAME_INCOME_TYPE],  # NAME_INCOME_TYPE
        EDUCATION_MAP[input_data.NAME_EDUCATION_TYPE],  # NAME_EDUCATION_TYPE
        FAMILY_STATUS_MAP[input_data.NAME_FAMILY_STATUS],  # NAME_FAMILY_STATUS
        HOUSING_TYPE_MAP[input_data.NAME_HOUSING_TYPE],  # NAME_HOUSING_TYPE
        0.02,  # REGION_POPULATION_RELATIVE
        -(input_data.AGE_YEARS * 365),  # DAYS_BIRTH
        -(input_data.YEARS_EMPLOYED * 365),  # DAYS_EMPLOYED
        -1000,  # DAYS_REGISTRATION
        -1000,  # DAYS_ID_PUBLISH
        1,  # FLAG_MOBIL
        1,  # FLAG_EMP_PHONE
        0,  # FLAG_WORK_PHONE
        1,  # FLAG_CONT_MOBILE
        0,  # FLAG_PHONE
        0,  # FLAG_EMAIL
        0,  # OCCUPATION_TYPE
        input_data.CNT_FAM_MEMBERS,  # CNT_FAM_MEMBERS
        input_data.REGION_RATING_CLIENT,  # REGION_RATING_CLIENT
        input_data.REGION_RATING_CLIENT,  # REGION_RATING_CLIENT_W_CITY
        0,  # WEEKDAY_APPR_PROCESS_START
        12,  # HOUR_APPR_PROCESS_START
        0,  # REG_REGION_NOT_LIVE_REGION
        0,  # REG_REGION_NOT_WORK_REGION
        0,  # LIVE_REGION_NOT_WORK_REGION
        0,  # REG_CITY_NOT_LIVE_CITY
        0,  # REG_CITY_NOT_WORK_CITY
        0,  # LIVE_CITY_NOT_WORK_CITY
        0,  # ORGANIZATION_TYPE
        input_data.EXT_SOURCE_2,  # EXT_SOURCE_2
        input_data.EXT_SOURCE_3,  # EXT_SOURCE_3
        0,  # OBS_30_CNT_SOCIAL_CIRCLE
        0,  # DEF_30_CNT_SOCIAL_CIRCLE
        0,  # OBS_60_CNT_SOCIAL_CIRCLE
        0,  # DEF_60_CNT_SOCIAL_CIRCLE
        0,  # DAYS_LAST_PHONE_CHANGE
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # FLAG_DOCUMENT_2 to 21
        0, 0, 0, 0, 0, 0,  # AMT_REQ_CREDIT_BUREAU_*
        input_data.AGE_YEARS,  # AGE_YEARS
        input_data.YEARS_EMPLOYED  # YEARS_EMPLOYED
    ]], columns=scaler.feature_names_in_)

    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = 1 if probability >= 0.5 else 0
    risk_level = get_risk_level(probability)

    return LoanPredictionOutput(
        default_probability=round(probability, 4),
        prediction=prediction,
        risk_level=risk_level
    )