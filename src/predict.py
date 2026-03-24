import joblib
import pandas as pd
import numpy as np

from preprocessing import preprocess_data


model = joblib.load("models/model.pkl")


def predict(data: dict):

    df = pd.DataFrame([data])

    df = preprocess_data(df)

    pred = model.predict(df)

    # Reverse log transform
    return np.expm1(pred[0])


if __name__ == "__main__":

    sample = {
        "MSSubClass": 60,
        "LotArea": 8450,
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "TotalBsmtSF": 856,
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "FullBath": 2,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 8,
        "GarageCars": 2,
        "GarageArea": 548,
        "YrSold": 2008
    }

    print("Predicted Price:", predict(sample))