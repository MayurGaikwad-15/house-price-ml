import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    # Feature Engineering
    df["TotalSF"] = (
        df.get("TotalBsmtSF", 0)
        + df.get("1stFlrSF", 0)
        + df.get("2ndFlrSF", 0)
    )

    df["HouseAge"] = df.get("YrSold", 0) - df.get("YearBuilt", 0)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("None")
        else:
            df[col] = df[col].fillna(0)

    return df