import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load model
# -------------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    layout="centered",
    page_icon="🏡"
)

# -------------------------------
# CSS for Apple-like UI
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #f5f5f7;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
h1, h2, h3, h4 {
    color: #111;
}
.stButton>button {
    background-color: #0071e3;
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.5em;
    font-size: 16px;
    font-weight: 500;
}
.stSlider>div>div>div>div>div>input {
    accent-color: #0071e3;
}
.stNumberInput>div>input {
    border-radius: 6px;
    border: 1px solid #ccc;
    padding: 6px 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title & Description
# -------------------------------
st.title("🏡 House Price Predictor")
st.markdown("Predict your dream house price in a few clicks ✨")

# -------------------------------
# Input fields in 3 columns
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    MSSubClass = st.number_input("MSSubClass", value=20)
    LotArea = st.number_input("LotArea", value=8000)
    OverallQual = st.slider("Overall Quality", 1, 10, 5)
    OverallCond = st.slider("Overall Condition", 1, 10, 5)

with col2:
    YearBuilt = st.number_input("Year Built", value=2000)
    TotalBsmtSF = st.number_input("Basement Area", value=800)
    FirstFlrSF = st.number_input("1st Floor Area", value=800)
    SecondFlrSF = st.number_input("2nd Floor Area", value=500)

with col3:
    FullBath = st.number_input("Full Bathrooms", value=2)
    BedroomAbvGr = st.number_input("Bedrooms", value=3)
    TotRmsAbvGrd = st.number_input("Total Rooms", value=6)
    GarageCars = st.number_input("Garage Cars", value=2)
    GarageArea = st.number_input("Garage Area", value=500)
    YrSold = st.number_input("Year Sold", value=2008)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("🚀 Predict Price"):
    # Prepare dataframe for prediction
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

    # Predict
    try:
        pred = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
