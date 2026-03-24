import streamlit as st
import requests

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction")

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
        "FirstFlrSF": FirstFlrSF,
        "SecondFlrSF": SecondFlrSF,
        "FullBath": FullBath,
        "BedroomAbvGr": BedroomAbvGr,
        "TotRmsAbvGrd": TotRmsAbvGrd,
        "GarageCars": GarageCars,
        "GarageArea": GarageArea,
        "YrSold": YrSold
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=data
    )

    if response.status_code == 200:
        result = response.json()
        if "price" in result:
            st.success(f"💰 Estimated Price: ${result['price']:,.2f}")
        else:
            st.error(result)
    else:
        st.error("API error")