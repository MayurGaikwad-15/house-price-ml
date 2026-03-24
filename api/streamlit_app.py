import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction")

# -----------------------------
# Load the trained pipeline
# -----------------------------
with open("models/model.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

# -----------------------------
# User Inputs
# -----------------------------
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

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Price"):
    input_data = pd.DataFrame([{
        "MSSubClass": MSSubClass,
        "LotArea": LotArea,
        "OverallQual": OverallQual,
        "OverallCond": OverallCond,
        "YearBuilt": YearBuilt,
        "TotalBsmtSF": TotalBsmtSF,
        "FirstFlrSF": FirstFlrSF,
        "SecondFlrSF": SecondFlrSF,
        "FullBath": FullBath,
        "BedroomAbvGr": BedroomAbvGr,
        "TotRmsAbvGrd": TotRmsAbvGrd,
        "GarageCars": GarageCars,
        "GarageArea": GarageArea,
        "YrSold": YrSold
    }])

    try:
        pred = model_pipeline.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
