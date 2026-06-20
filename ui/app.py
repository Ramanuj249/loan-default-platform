import streamlit as st
import requests


st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦")
st.title("Loan Default Prediction System")
st.write("Enter customer details to predict loan default probability")

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income", min_value=0, value=150000)
    credit_amount = st.number_input("Loan Amount", min_value=0, value=300000)
    annuity = st.number_input("Annual Payment", min_value=0, value=15000)
    goods_price = st.number_input("Goods Price", min_value=0, value=250000)
    children = st.number_input("Number of Children", min_value=0, value=0)
    fam_members = st.number_input("Family Members", min_value=1, value=2)
    region_rating = st.selectbox("Region Rating", [1, 2, 3])

with col2:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    years_employed = st.number_input("Years Employed", min_value=0, value=5)
    # ext_source_1 = st.slider("Credit Score 1 (External)", 0.0, 1.0, 0.5)
    ext_source_2 = st.slider("Credit Score 2 (External)", 0.0, 1.0, 0.5)
    ext_source_3 = st.slider("Credit Score 3 (External)", 0.0, 1.0, 0.5)

st.subheader("Personal & Loan Details")

col3, col4 = st.columns(2)

with col3:
    gender = st.selectbox("Gender", ["F", "M"])
    education = st.selectbox("Education", ["Academic degree", "Higher education", "Incomplete higher", "Lower secondary", "Secondary / secondary special"])
    family_status = st.selectbox("Family Status", ["Civil marriage", "Married", "Separated", "Single / not married", "Widow"])
    housing_type = st.selectbox("Housing Type", ["Co-op apartment", "House / apartment", "Municipal apartment", "Office apartment", "Rented apartment", "With parents"])

with col4:
    income_type = st.selectbox("Income Type", ["Businessman", "Commercial associate", "Pensioner", "State servant", "Working"])
    contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
    own_car = st.selectbox("Owns Car", ["Yes", "No"])
    own_realty = st.selectbox("Owns Property", ["Yes", "No"])


if st.button("Predict Default Risk"):
    # Prepare data to send to API
    own_car_value = "Y" if own_car == "Yes" else "N"
    own_realty_value = "Y" if own_realty == "Yes" else "N"
    data = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit_amount,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "CNT_CHILDREN": children,
        "CNT_FAM_MEMBERS": fam_members,
        "AGE_YEARS": age,
        "YEARS_EMPLOYED": years_employed,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "CODE_GENDER": gender,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "NAME_INCOME_TYPE": income_type,
        "NAME_CONTRACT_TYPE": contract_type,
        "FLAG_OWN_CAR": own_car_value,
        "FLAG_OWN_REALTY": own_realty_value,
        "REGION_RATING_CLIENT": region_rating
    }

    # Call FastAPI
    response = requests.post("http://localhost:8000/predict", json=data)

    if response.status_code == 200:
        result = response.json()

        st.subheader("Prediction Result")
        st.write(f"Default Probability: {result['default_probability'] * 100:.2f}%")
        st.write(f"Risk Level: {result['risk_level']}")

        if result['prediction'] == 1:
            st.error("High risk of default")
        else:
            st.success("Low risk of default")
    else:
        st.error("Error connecting to API")