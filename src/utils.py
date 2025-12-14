# src/utils.py
"""
Utility functions for the Real Estate Investment Advisor project:
- Formatting helpers
- Investment description
- Feature importance
- EDA plotting helpers (for Streamlit or notebooks)
"""

from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


# ==============================
# Formatting & interpretation
# ==============================

def format_price_lakhs(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"₹ {value:,.2f} Lakhs"


def format_percentage(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f} %"


def compute_growth_percentage(current_price_lakhs: float, future_price_lakhs: float) -> float:
    if current_price_lakhs <= 0:
        return np.nan
    return (future_price_lakhs - current_price_lakhs) / current_price_lakhs


def describe_investment_label(label: int, prob_good: float | None = None) -> str:
    if label == 1:
        if prob_good is not None:
            return f"Good Investment (confidence: {prob_good:.2%})"
        return "Good Investment"
    else:
        if prob_good is not None:
            return f"Not a Recommended Investment (probability of good: {prob_good:.2%})"
        return "Not a Recommended Investment"


# ==============================
# Feature importance utils
# ==============================

def get_feature_names_from_pipeline(pipe: Pipeline) -> List[str]:
    if "preprocessor" not in pipe.named_steps:
        raise ValueError("Pipeline must contain a 'preprocessor' step.")

    preprocessor = pipe.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback: generic features
        n_features = pipe.named_steps["model"].n_features_in_
        feature_names = [f"feature_{i}" for i in range(n_features)]

    return list(feature_names)


def get_tree_feature_importance(pipe: Pipeline, top_n: int = 20) -> pd.DataFrame:
    if "model" not in pipe.named_steps:
        raise ValueError("Pipeline must contain a 'model' step.")

    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose 'feature_importances_'.")

    feature_names = get_feature_names_from_pipeline(pipe)
    importances = model.feature_importances_

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return fi.head(top_n)


def get_permutation_importance(
    pipe: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scoring: str = "r2",
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 20,
) -> pd.DataFrame:
    result = permutation_importance(
        pipe,
        X_valid,
        y_valid,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    feature_names = get_feature_names_from_pipeline(pipe)

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return df_imp.head(top_n)


# ==============================
# EDA plotting helpers
# ==============================

def plot_price_distribution(df: pd.DataFrame, column: str = "Price_in_Lakhs", bins: int = 40) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=bins)
    ax.set_title("Distribution of Property Prices")
    ax.set_xlabel("Price (Lakhs)")
    ax.set_ylabel("Count")
    return fig


def plot_size_distribution(df: pd.DataFrame, column: str = "Size_in_SqFt", bins: int = 40) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=bins)
    ax.set_title("Distribution of Property Sizes")
    ax.set_xlabel("Size (SqFt)")
    ax.set_ylabel("Count")
    return fig


def plot_price_vs_size(
    df: pd.DataFrame,
    size_col: str = "Size_in_SqFt",
    price_col: str = "Price_in_Lakhs",
    sample_n: int = 2000,
) -> plt.Figure:
    if len(df) > sample_n:
        sample = df[[size_col, price_col]].dropna().sample(sample_n, random_state=42)
    else:
        sample = df[[size_col, price_col]].dropna()

    fig, ax = plt.subplots()
    ax.scatter(sample[size_col], sample[price_col], alpha=0.5)
    ax.set_title("Property Size vs Price")
    ax.set_xlabel("Size (SqFt)")
    ax.set_ylabel("Price (Lakhs)")
    return fig


def plot_price_per_sqft_by_city(
    df: pd.DataFrame,
    city_col: str = "City",
    ppsf_col: str = "Price_per_SqFt",
    top_n: int = 10,
) -> plt.Figure:
    grouped = (
        df.groupby(city_col)[ppsf_col]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots()
    grouped.plot(kind="bar", ax=ax)
    ax.set_title(f"Top {top_n} Cities by Avg Price per SqFt")
    ax.set_ylabel("Avg Price per SqFt")
    ax.set_xlabel("City")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_public_transport_vs_ppsf(
    df: pd.DataFrame,
    transport_col: str = "Public_Transport_Accessibility",
    ppsf_col: str = "Price_per_SqFt",
) -> plt.Figure:
    # convert High/Medium/Low to ordered numeric for plotting
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    temp = df.copy()
    temp["pta_num"] = temp[transport_col].map(mapping)

    grouped = temp.groupby("pta_num")[ppsf_col].mean().sort_index()
    labels = {1: "Low", 2: "Medium", 3: "High"}

    fig, ax = plt.subplots()
    grouped.plot(kind="line", marker="o", ax=ax)
    ax.set_xticks(list(labels.keys()))
    ax.set_xticklabels([labels[i] for i in labels.keys()])
    ax.set_title("Public Transport Accessibility vs Price per SqFt")
    ax.set_xlabel("Accessibility")
    ax.set_ylabel("Avg Price per SqFt")
    return fig
