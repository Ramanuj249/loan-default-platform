from fastapi import FastAPI

from api.schemas import LoanPredictionInput, LoanPredictionOutput
from api.model_loader import load_model, get_risk_level
import numpy as np
from contextlib import asynccontextmanager

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
    # Convert input to numpy array
    features = np.array([[
        input_data.SK_ID_CURR,
        input_data.NAME_CONTRACT_TYPE,
        input_data.CODE_GENDER,
        input_data.FLAG_OWN_CAR,
        input_data.FLAG_OWN_REALTY,
        input_data.CNT_CHILDREN,
        input_data.AMT_INCOME_TOTAL,
        input_data.AMT_CREDIT,
        input_data.AMT_ANNUITY,
        input_data.AMT_GOODS_PRICE,
        input_data.NAME_TYPE_SUITE,
        input_data.NAME_INCOME_TYPE,
        input_data.NAME_EDUCATION_TYPE,
        input_data.NAME_FAMILY_STATUS,
        input_data.NAME_HOUSING_TYPE,
        input_data.REGION_POPULATION_RELATIVE,
        input_data.DAYS_BIRTH,
        input_data.DAYS_EMPLOYED,
        input_data.DAYS_REGISTRATION,
        input_data.DAYS_ID_PUBLISH,
        input_data.FLAG_MOBIL,
        input_data.FLAG_EMP_PHONE,
        input_data.FLAG_WORK_PHONE,
        input_data.FLAG_CONT_MOBILE,
        input_data.FLAG_PHONE,
        input_data.FLAG_EMAIL,
        input_data.OCCUPATION_TYPE,
        input_data.CNT_FAM_MEMBERS,
        input_data.REGION_RATING_CLIENT,
        input_data.REGION_RATING_CLIENT_W_CITY,
        input_data.WEEKDAY_APPR_PROCESS_START,
        input_data.HOUR_APPR_PROCESS_START,
        input_data.REG_REGION_NOT_LIVE_REGION,
        input_data.REG_REGION_NOT_WORK_REGION,
        input_data.LIVE_REGION_NOT_WORK_REGION,
        input_data.REG_CITY_NOT_LIVE_CITY,
        input_data.REG_CITY_NOT_WORK_CITY,
        input_data.LIVE_CITY_NOT_WORK_CITY,
        input_data.ORGANIZATION_TYPE,
        input_data.EXT_SOURCE_2,
        input_data.EXT_SOURCE_3,
        input_data.OBS_30_CNT_SOCIAL_CIRCLE,
        input_data.DEF_30_CNT_SOCIAL_CIRCLE,
        input_data.OBS_60_CNT_SOCIAL_CIRCLE,
        input_data.DEF_60_CNT_SOCIAL_CIRCLE,
        input_data.DAYS_LAST_PHONE_CHANGE,
        input_data.FLAG_DOCUMENT_2,
        input_data.FLAG_DOCUMENT_3,
        input_data.FLAG_DOCUMENT_4,
        input_data.FLAG_DOCUMENT_5,
        input_data.FLAG_DOCUMENT_6,
        input_data.FLAG_DOCUMENT_7,
        input_data.FLAG_DOCUMENT_8,
        input_data.FLAG_DOCUMENT_9,
        input_data.FLAG_DOCUMENT_10,
        input_data.FLAG_DOCUMENT_11,
        input_data.FLAG_DOCUMENT_12,
        input_data.FLAG_DOCUMENT_13,
        input_data.FLAG_DOCUMENT_14,
        input_data.FLAG_DOCUMENT_15,
        input_data.FLAG_DOCUMENT_16,
        input_data.FLAG_DOCUMENT_17,
        input_data.FLAG_DOCUMENT_18,
        input_data.FLAG_DOCUMENT_19,
        input_data.FLAG_DOCUMENT_20,
        input_data.FLAG_DOCUMENT_21,
        input_data.AMT_REQ_CREDIT_BUREAU_HOUR,
        input_data.AMT_REQ_CREDIT_BUREAU_DAY,
        input_data.AMT_REQ_CREDIT_BUREAU_WEEK,
        input_data.AMT_REQ_CREDIT_BUREAU_MON,
        input_data.AMT_REQ_CREDIT_BUREAU_QRT,
        input_data.AMT_REQ_CREDIT_BUREAU_YEAR,
        input_data.AGE_YEARS,
        input_data.YEARS_EMPLOYED,
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Get prediction
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = 1 if probability >= 0.5 else 0
    risk_level = get_risk_level(probability)

    return LoanPredictionOutput(
        default_probability=round(probability, 4),
        prediction=prediction,
        risk_level=risk_level
    )