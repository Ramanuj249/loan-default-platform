from pydantic import BaseModel


class LoanPredictionInput(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    CNT_CHILDREN: int
    CNT_FAM_MEMBERS: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    CODE_GENDER: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    NAME_INCOME_TYPE: str
    NAME_CONTRACT_TYPE: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    REGION_RATING_CLIENT: int
    AGE_YEARS: float
    YEARS_EMPLOYED: float


class LoanPredictionOutput(BaseModel):
    default_probability: float
    prediction: int
    risk_level: str