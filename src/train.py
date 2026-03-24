import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.preprocessing import preprocess_data
from src.pipeline import build_pipeline


def train():

    print("📥 Loading data...")
    df = pd.read_csv("data/train.csv")

    y = np.log1p(df["SalePrice"])
    X = df.drop("SalePrice", axis=1)

    print("🧹 Preprocessing data...")
    X = preprocess_data(X)

    print("✂️ Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, random_state=42)
    }

    best_model = None
    best_score = float("inf")
    best_pipeline = None

    print("🤖 Training models...\n")

    for name, model in models.items():

        print(f"🔹 Training {name}...")

        pipeline, num_cols, cat_cols = build_pipeline(model, X_train)

        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

        print(f"{name} Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model
            best_pipeline = pipeline

    print("\n🏆 Best model selected!")
    print(f"Best RMSE: {best_score:.4f}")

    # ✅ SAVE EVERYTHING
    joblib.dump({
        "pipeline": best_pipeline,
        "columns": X.columns.tolist()
    }, "models/model.pkl")

    print("✅ Model + schema saved!")


if __name__ == "__main__":
    train()