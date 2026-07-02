import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction")

# Load model directly instead of calling API
@st.cache_resource
def load_model():
    artifact = joblib.load("models/model.pkl")
    return artifact["pipeline"], artifact["columns"]

pipeline, train_columns = load_model()

MSSubClass = st.number_input("MSSubClass", value=20)
LotArea = st.number_input("LotArea", value=8000)
OverallQual = st.slider("Overall Quality", 1, 10, 5)
OverallCond = st.slider("Overall Condition", 1, 10, 5)
YearBuilt = st.number_input("Year Built", value=2000)
TotalBsmtSF = st.number_input("Basement Area", value=800)
FirstFlrSF = st.number_input("1st Floor Area", value=800)
SecondFlrSF = st.number_input("2nd Floor Area", value=500)
FullBath = st.number_input("Full Bathrooms", value=2)
BedroomAbvGr = st.number_input("Bedrooms", value=3)
TotRmsAbvGrd = st.number_input("Total Rooms", value=6)
GarageCars = st.number_input("Garage Cars", value=2)
GarageArea = st.number_input("Garage Area", value=500)
YrSold = st.number_input("Year Sold", value=2008)

if st.button("🚀 Predict Price"):

    data = {
        "MSSubClass": MSSubClass,
        "LotArea": LotArea,
        "OverallQual": OverallQual,
        "OverallCond": OverallCond,
        "YearBuilt": YearBuilt,
        "TotalBsmtSF": TotalBsmtSF,
        "1stFlrSF": FirstFlrSF,
        "2ndFlrSF": SecondFlrSF,
        "FullBath": FullBath,
        "BedroomAbvGr": BedroomAbvGr,
        "TotRmsAbvGrd": TotRmsAbvGrd,
        "GarageCars": GarageCars,
        "GarageArea": GarageArea,
        "YrSold": YrSold
    }

    try:
        df = pd.DataFrame([data])

        # Add missing columns
        for col in train_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[train_columns]

        pred = pipeline.predict(df)[0]
        price = float(np.expm1(pred))

        st.success(f"💰 Estimated Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
