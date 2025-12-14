# src/train_models.py
"""
FAST training script (no MLflow) for Real Estate Investment Advisor

Trains:
- RandomForestClassifier -> Good_Investment
- RandomForestRegressor  -> Future_Price_5Y

Saves:
- models/classifier.pkl
- models/regressor.pkl
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import joblib

DATA_PATH = Path("data/processed_housing.csv")
MODELS_DIR = Path("models")


def load_processed() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Use the same features that the Streamlit app sends.
    """
    feature_cols = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Furnished_Status",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Amenities",
        "Facing",
        "Owner_Type",
        "Availability_Status",
    ]

    X = df[feature_cols].copy()
    return X, feature_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor


def train_classifier(df: pd.DataFrame, preprocessor: ColumnTransformer, feature_cols):
    print("▶ Training classifier (Good_Investment)...")
    X = df[feature_cols]
    y = df["Good_Investment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=50,        # fewer trees = faster
        max_depth=12,          # limit depth
        min_samples_leaf=5,    # avoids overfitting, speeds up
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Classifier Accuracy: {acc:.4f}")
    print(f"✅ Classifier F1-score: {f1:.4f}")

    return pipe


def train_regressor(df: pd.DataFrame, preprocessor: ColumnTransformer, feature_cols):
    print("▶ Training regressor (Future_Price_5Y)...")
    X = df[feature_cols]
    y = df["Future_Price_5Y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", reg),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Your sklearn version doesn't support squared=False, so compute RMSE manually
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Regressor RMSE: {rmse:.4f}")
    print(f"✅ Regressor MAE : {mae:.4f}")
    print(f"✅ Regressor R²  : {r2:.4f}")

    return pipe


def main():
    df = load_processed()
    X, feature_cols = build_feature_matrix(df)
    preprocessor = build_preprocessor(X)

    MODELS_DIR.mkdir(exist_ok=True)

    clf_pipe = train_classifier(df, preprocessor, feature_cols)
    joblib.dump(clf_pipe, MODELS_DIR / "classifier.pkl")

    reg_pipe = train_regressor(df, preprocessor, feature_cols)
    joblib.dump(reg_pipe, MODELS_DIR / "regressor.pkl")

    print(f"✅ Models saved in {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
