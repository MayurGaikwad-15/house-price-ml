from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

artifact = joblib.load("models/model.pkl")
pipeline = artifact["pipeline"]
train_columns = artifact["columns"]


@app.get("/")
def home():
    return {"message": "🚀 API running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    df.rename(columns={
        "FirstFlrSF": "1stFlrSF",
        "SecondFlrSF": "2ndFlrSF"
    }, inplace=True)

    # Add missing columns
    for col in train_columns:
        if col not in df.columns:
            df[col] = np.nan   # ✅ keep NaN

    df = df[train_columns]

    pred = pipeline.predict(df)[0]

    return {"price": float(np.expm1(pred))}