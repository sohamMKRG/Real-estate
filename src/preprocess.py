# src/preprocess.py
"""
Preprocess india_housing_prices.csv:
- Basic cleaning
- Feature engineering (growth, future price)
- Create Good_Investment label
- Save processed_housing.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/india_housing_prices.csv")
PROCESSED_PATH = Path("data/processed_housing.csv")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, trim strings. (Your dataset has no nulls already.)"""
    # Drop exact duplicates
    df = df.drop_duplicates()

    # Strip spaces from string columns
    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols) > 0:
        df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create extra features + targets:

    - Price_per_SqFt (if missing)
    - Age_of_Property (if missing)
    - Nearby_Schools_Score, Nearby_Hospitals_Score
    - Growth_Rate (by city) + Future_Price_5Y
    - Good_Investment (0/1) using rule-based score
    """

    # -------------------------------------------------
    # Price per SqFt (already present in your data, but keep for safety)
    # -------------------------------------------------
    if "Price_per_SqFt" not in df.columns:
        df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]

    # -------------------------------------------------
    # Age of property (already present, but compute if missing)
    # -------------------------------------------------
    if "Age_of_Property" not in df.columns and "Year_Built" in df.columns:
        CURRENT_YEAR = 2024
        df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]

    # -------------------------------------------------
    # School & hospital density scores (0–1 normalized)
    # -------------------------------------------------
    for col in ["Nearby_Schools", "Nearby_Hospitals"]:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            df[col + "_Score"] = (df[col] - col_min) / (col_max - col_min + 1e-6)

    # -------------------------------------------------
    # City-based growth rate (simple demo logic)
    # -------------------------------------------------
    city_growth = {
        "Mumbai": 0.10,
        "Delhi": 0.09,
        "Bengaluru": 0.11,
        "Hyderabad": 0.105,
        "Pune": 0.095,
    }
    default_growth = 0.08

    df["Growth_Rate"] = df["City"].map(city_growth).fillna(default_growth)
    YEARS = 5

    # Future price in Lakhs (compound)
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** YEARS

    # Appreciation %
    df["Appreciation_Rate_5Y"] = (
        df["Future_Price_5Y"] - df["Price_in_Lakhs"]
    ) / df["Price_in_Lakhs"]

    # -------------------------------------------------
    # City median price per sqft
    # -------------------------------------------------
    df["City_Median_PPSF"] = df.groupby("City")["Price_per_SqFt"].transform("median")
    df["Below_Median_PPSF"] = df["Price_per_SqFt"] <= df["City_Median_PPSF"]

    # -------------------------------------------------
    # RULES for Good_Investment
    # -------------------------------------------------
    appreciation_threshold = 0.40  # ≥40% in 5 yrs

    rule_appreciation = df["Appreciation_Rate_5Y"] >= appreciation_threshold
    rule_ppsf = df["Below_Median_PPSF"]  # cheaper than city median
    rule_bhk = df["BHK"] >= 2

    # ---------- Public Transport (High / Medium / Low) ----------
    if "Public_Transport_Accessibility" in df.columns:
        pta = df["Public_Transport_Accessibility"]

        # Map categories like Low/Medium/High -> 1/2/3
        mapping = {"Low": 1, "Medium": 2, "High": 3}
        pta_num = pta.map(mapping).fillna(2)  # default Medium

        median_pta = pta_num.median()
        rule_transport = pta_num >= median_pta
    else:
        rule_transport = pd.Series(True, index=df.index)

    # ---------- Parking (Yes / No) ----------
    if "Parking_Space" in df.columns:
        ps_raw = df["Parking_Space"].astype(str).str.lower()

        # convert "yes"/"no" to 1/0
        yes_mask = ps_raw.isin(["yes", "y", "available", "present", "included", "parking"])
        no_mask = ps_raw.isin(["no", "n", "none", "na", "n/a", "zero"])

        ps_num = pd.to_numeric(df["Parking_Space"], errors="coerce")
        ps_num = ps_num.where(~yes_mask, 1)
        ps_num = ps_num.where(~no_mask, 0)
        ps_num = ps_num.fillna(0)

        rule_parking = ps_num > 0
    else:
        rule_parking = pd.Series(True, index=df.index)

    # -------------------------------------------------
    # Combine into Investment_Score
    # -------------------------------------------------
    investment_score = (
        rule_appreciation.astype(int)
        + rule_ppsf.astype(int)
        + rule_bhk.astype(int)
        + rule_transport.astype(int)
        + rule_parking.astype(int)
    )
    df["Investment_Score"] = investment_score

    # Threshold: 3 or more satisfied = Good Investment
    df["Good_Investment"] = (df["Investment_Score"] >= 3).astype(int)

    return df


def main():
    df = load_data()
    df = basic_cleaning(df)
    df = feature_engineering(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Saved processed data to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
