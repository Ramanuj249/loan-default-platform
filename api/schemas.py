from pydantic import BaseModel
from typing import Optional

class LoanPredictionInput(BaseModel):
    SK_ID_CURR: Optional[float] = 0
    NAME_CONTRACT_TYPE: Optional[float] = 0
    CODE_GENDER: Optional[float] = 0
    FLAG_OWN_CAR: Optional[float] = 0
    FLAG_OWN_REALTY: Optional[float] = 0
    CNT_CHILDREN: Optional[float] = 0
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: Optional[float] = 0
    NAME_TYPE_SUITE: Optional[float] = 0
    NAME_INCOME_TYPE: Optional[float] = 0
    NAME_EDUCATION_TYPE: Optional[float] = 0
    NAME_FAMILY_STATUS: Optional[float] = 0
    NAME_HOUSING_TYPE: Optional[float] = 0
    REGION_POPULATION_RELATIVE: Optional[float] = 0
    DAYS_BIRTH: Optional[float] = 0
    DAYS_EMPLOYED: Optional[float] = 0
    DAYS_REGISTRATION: Optional[float] = 0
    DAYS_ID_PUBLISH: Optional[float] = 0
    FLAG_MOBIL: Optional[float] = 0
    FLAG_EMP_PHONE: Optional[float] = 0
    FLAG_WORK_PHONE: Optional[float] = 0
    FLAG_CONT_MOBILE: Optional[float] = 0
    FLAG_PHONE: Optional[float] = 0
    FLAG_EMAIL: Optional[float] = 0
    OCCUPATION_TYPE: Optional[float] = 0
    CNT_FAM_MEMBERS: Optional[float] = 0
    REGION_RATING_CLIENT: Optional[float] = 2
    REGION_RATING_CLIENT_W_CITY: Optional[float] = 2
    WEEKDAY_APPR_PROCESS_START: Optional[float] = 0
    HOUR_APPR_PROCESS_START: Optional[float] = 12
    REG_REGION_NOT_LIVE_REGION: Optional[float] = 0
    REG_REGION_NOT_WORK_REGION: Optional[float] = 0
    LIVE_REGION_NOT_WORK_REGION: Optional[float] = 0
    REG_CITY_NOT_LIVE_CITY: Optional[float] = 0
    REG_CITY_NOT_WORK_CITY: Optional[float] = 0
    LIVE_CITY_NOT_WORK_CITY: Optional[float] = 0
    ORGANIZATION_TYPE: Optional[float] = 0
    EXT_SOURCE_2: Optional[float] = 0.5
    EXT_SOURCE_3: Optional[float] = 0.5
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = 0
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = 0
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = 0
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = 0
    DAYS_LAST_PHONE_CHANGE: Optional[float] = 0
    FLAG_DOCUMENT_2: Optional[float] = 0
    FLAG_DOCUMENT_3: Optional[float] = 1
    FLAG_DOCUMENT_4: Optional[float] = 0
    FLAG_DOCUMENT_5: Optional[float] = 0
    FLAG_DOCUMENT_6: Optional[float] = 0
    FLAG_DOCUMENT_7: Optional[float] = 0
    FLAG_DOCUMENT_8: Optional[float] = 0
    FLAG_DOCUMENT_9: Optional[float] = 0
    FLAG_DOCUMENT_10: Optional[float] = 0
    FLAG_DOCUMENT_11: Optional[float] = 0
    FLAG_DOCUMENT_12: Optional[float] = 0
    FLAG_DOCUMENT_13: Optional[float] = 0
    FLAG_DOCUMENT_14: Optional[float] = 0
    FLAG_DOCUMENT_15: Optional[float] = 0
    FLAG_DOCUMENT_16: Optional[float] = 0
    FLAG_DOCUMENT_17: Optional[float] = 0
    FLAG_DOCUMENT_18: Optional[float] = 0
    FLAG_DOCUMENT_19: Optional[float] = 0
    FLAG_DOCUMENT_20: Optional[float] = 0
    FLAG_DOCUMENT_21: Optional[float] = 0
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = 0
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = 0
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = 0
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = 0
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = 0
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = 0
    AGE_YEARS: float
    YEARS_EMPLOYED: float

class LoanPredictionOutput(BaseModel):
    default_probability: float
    prediction: int
    risk_level:str
