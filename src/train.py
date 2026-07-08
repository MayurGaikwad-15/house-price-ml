import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.preprocessing import preprocess_data
from src.pipeline import build_pipeline


def train():

    print("Loading data...")
    df = pd.read_csv("data/train.csv")

    y = np.log1p(df["SalePrice"])
    X = df.drop("SalePrice", axis=1)

    print("Preprocessing data...")
    X = preprocess_data(X)

    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge":        Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost":      XGBRegressor(
                            n_estimators=300,
                            learning_rate=0.05,
                            max_depth=5,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            verbosity=0,
                        ),
    }

    best_model_name = None
    best_score = float("inf")
    best_pipeline = None
    results = {}

    print("Training models...\n")

    for name, model in models.items():
        print(f"  Training {name}...")

        pipeline, num_cols, cat_cols = build_pipeline(model, X_train)
        pipeline.fit(X_train, y_train)

        # Validation metrics
        val_preds = pipeline.predict(X_val)
        val_rmse  = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        val_r2    = float(r2_score(y_val, val_preds))

        # 5-fold CV RMSE on full training split
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=5, scoring="neg_root_mean_squared_error"
        )
        cv_rmse = float(-cv_scores.mean())
        cv_std  = float(cv_scores.std())

        print(f"  {name}: Val RMSE={val_rmse:.4f}  R2={val_r2:.4f}  CV RMSE={cv_rmse:.4f}+-{cv_std:.4f}")

        results[name] = {
            "val_rmse": round(val_rmse, 4),
            "val_r2":   round(val_r2,   4),
            "cv_rmse":  round(cv_rmse,  4),
            "cv_std":   round(cv_std,   4),
        }

        if val_rmse < best_score:
            best_score      = val_rmse
            best_model_name = name
            best_pipeline   = pipeline

    print(f"\nBest model: {best_model_name}  (RMSE={best_score:.4f})")

    # Mark winner
    for name in results:
        results[name]["best"] = (name == best_model_name)

    # Save model + schema
    # Also persist the transformed X_train matrix so explain.py can run SHAP
    # without re-fitting the preprocessor.
    preprocessor  = best_pipeline.named_steps["preprocessor"]
    X_train_trans = preprocessor.transform(X_train)

    # Recover feature names after one-hot encoding
    num_features = best_pipeline.named_steps["preprocessor"] \
                     .transformers_[0][2]          # numeric col names
    ohe          = best_pipeline.named_steps["preprocessor"] \
                     .named_transformers_["cat"]    \
                     .named_steps["onehot"]
    cat_features = list(ohe.get_feature_names_out(
        best_pipeline.named_steps["preprocessor"].transformers_[1][2]
    ))
    feature_names = list(num_features) + cat_features

    joblib.dump(
        {
            "pipeline":        best_pipeline,
            "columns":         X.columns.tolist(),
            "X_train_trans":   X_train_trans,
            "feature_names":   feature_names,
            "best_model_name": best_model_name,
        },
        "models/model.pkl",
    )

    # Save results table
    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Model + schema + metrics saved!")


if __name__ == "__main__":
    train()