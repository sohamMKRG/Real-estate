# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils import (
    format_price_lakhs,
    format_percentage,
    compute_growth_percentage,
    describe_investment_label,
    plot_price_distribution,
    plot_size_distribution,
    plot_price_vs_size,
    plot_price_per_sqft_by_city,
    plot_public_transport_vs_ppsf,
    get_tree_feature_importance,
)

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
    page_icon="🏠",
)

DATA_PATH = Path("data/processed_housing_small.csv")
CLASSIFIER_PATH = Path("models/classifier.pkl")
REGRESSOR_PATH = Path("models/regressor.pkl")


# ---------------- LOAD DATA & MODELS ----------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    clf = joblib.load(CLASSIFIER_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    return clf, reg


df = load_data()
clf, reg = load_models()

# Precompute things needed for rule-based score
pta_mapping = {"Low": 1, "Medium": 2, "High": 3}
df_pta_num = df["Public_Transport_Accessibility"].map(pta_mapping).fillna(2)
median_pta_num_global = df_pta_num.median()

# ---------------- HEADER ----------------
st.title("🏠 Real Estate Investment Advisor")
st.markdown(
    """
Use this app to:
- Classify whether a property is a **Good Investment**
- Predict its **Estimated Price after 5 years**

Models: Random Forest (Classification + Regression)  
Dataset: Indian Housing Prices  
"""
)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("⚙️ Property Details")

def get_unique(col: str):
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())
    return []

state_list = get_unique("State")
city_list = get_unique("City")
prop_types = get_unique("Property_Type")
furn_list = get_unique("Furnished_Status")
sec_list = get_unique("Security")
face_list = get_unique("Facing")
owner_list = get_unique("Owner_Type")
avail_list = get_unique("Availability_Status")
amen_list = get_unique("Amenities")

city = st.sidebar.selectbox("City", city_list)

# Locality options based on selected city
if "Locality" in df.columns:
    loc_opts = df[df["City"] == city]["Locality"].unique().tolist()
    if not loc_opts:
        loc_opts = df["Locality"].unique().tolist()
    locality = st.sidebar.selectbox("Locality", sorted(loc_opts))
else:
    locality = st.sidebar.text_input("Locality", "Unknown")

# Infer default State from city
if "State" in df.columns:
    city_states = df[df["City"] == city]["State"]
    if not city_states.empty:
        default_state = city_states.mode().iloc[0]
    else:
        default_state = df["State"].mode().iloc[0]
else:
    default_state = "Unknown"

state_index = state_list.index(default_state) if default_state in state_list else 0
state = st.sidebar.selectbox("State", state_list, index=state_index)

property_type = st.sidebar.selectbox("Property Type", prop_types)
bhk = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=3, step=1)
size_sqft = st.sidebar.number_input(
    "Size (SqFt)", min_value=200, max_value=15000, value=1200, step=50
)

price_lakhs = st.sidebar.number_input(
    "Current Price (Lakhs)",
    min_value=5.0,
    max_value=5000.0,
    value=80.0,
    step=1.0,
    help="Enter the property's current market price in Lakhs. "
         "It is used to compute price per SqFt, future price and profit.",
)

furn_status = st.sidebar.selectbox("Furnished Status", furn_list)
floor_no = st.sidebar.number_input("Floor No", min_value=0, max_value=200, value=2, step=1)
total_floors = st.sidebar.number_input("Total Floors", min_value=1, max_value=200, value=10, step=1)
age_prop = st.sidebar.number_input("Age of Property (years)", min_value=0, max_value=100, value=5, step=1)

nearby_schools = st.sidebar.number_input("Nearby Schools", min_value=0, max_value=50, value=3, step=1)
nearby_hospitals = st.sidebar.number_input("Nearby Hospitals", min_value=0, max_value=50, value=2, step=1)
public_transport = st.sidebar.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"])
parking = st.sidebar.selectbox("Parking Space", ["No", "Yes"])

security = st.sidebar.selectbox("Security", sec_list)
facing = st.sidebar.selectbox("Facing", face_list)
owner_type = st.sidebar.selectbox("Owner Type", owner_list)
availability = st.sidebar.selectbox("Availability Status", avail_list)
amenities = st.sidebar.selectbox("Amenities", amen_list)

# Derived: Price per SqFt
price_per_sqft = (price_lakhs * 1e5) / size_sqft

input_dict = {
    "State": state,
    "City": city,
    "Locality": locality,
    "Property_Type": property_type,
    "BHK": bhk,
    "Size_in_SqFt": size_sqft,
    "Price_in_Lakhs": price_lakhs,
    "Price_per_SqFt": price_per_sqft,
    "Furnished_Status": furn_status,
    "Floor_No": floor_no,
    "Total_Floors": total_floors,
    "Age_of_Property": age_prop,
    "Nearby_Schools": nearby_schools,
    "Nearby_Hospitals": nearby_hospitals,
    "Public_Transport_Accessibility": public_transport,
    "Parking_Space": parking,
    "Security": security,
    "Amenities": amenities,
    "Facing": facing,
    "Owner_Type": owner_type,
    "Availability_Status": availability,
}

input_df = pd.DataFrame([input_dict])

# ---------------- TABS ----------------
tab_pred, tab_eda, tab_feat = st.tabs(
    ["🔮 Predictions", "📊 EDA & Insights", "📈 Feature Importance"]
)

# ---- TAB 1: PREDICTION ----
with tab_pred:
    st.subheader("🔮 Investment Prediction & Price Forecast")

    st.write("### Input Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

    if st.button("🚀 Predict", type="primary"):
        # --- Model predictions ---
        pred_class = int(clf.predict(input_df)[0])
        proba_good = float(clf.predict_proba(input_df)[0][1])
        future_price_5y = float(reg.predict(input_df)[0])

        # Current price, growth & profit
        growth_pct = compute_growth_percentage(price_lakhs, future_price_5y)
        profit_lakhs = future_price_5y - price_lakhs

        # --- Top summary metrics ---
        st.markdown("### 📈 Investment Forecast Results")

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric(
                label="Current Price",
                value=format_price_lakhs(price_lakhs),
            )
        with mcol2:
            st.metric(
                label="Estimated Price in 5 Years",
                value=format_price_lakhs(future_price_5y),
                delta=format_price_lakhs(profit_lakhs),
                delta_color="normal",
            )
        with mcol3:
            st.metric(
                label="Expected Appreciation",
                value=format_percentage(growth_pct),
            )

        # --- Classification result with confidence ---
        col1, col2 = st.columns(2)

        with col1:
            if pred_class == 1:
                st.success("✅ GOOD INVESTMENT predicted")
            else:
                st.error("⚠️ NOT A RECOMMENDED INVESTMENT")

            st.write(
                f"**Classifier Output:** {describe_investment_label(pred_class, proba_good)}"
            )

        with col2:
            # Small comparison bar chart: current vs future price
            st.write("**Price Growth Comparison (Lakhs)**")
            comp_df = pd.DataFrame(
                {
                    "Price (Lakhs)": [price_lakhs, future_price_5y],
                },
                index=["Current", "5 Years (Predicted)"],
            )
            st.bar_chart(comp_df)

        st.markdown("---")
        st.write("### Model Confidence")
        st.progress(proba_good)
        st.caption("Progress bar shows model confidence for 'Good Investment' class.")

        # ============================
        # 🧮 RULE-BASED INVESTMENT SCORE BREAKDOWN
        # (Same logic that was used to create Good_Investment label)
        # ============================
        st.markdown("### 🧮 Rule-based Investment Score (used to label training data)")

        # 1. Appreciation condition (using predicted appreciation here)
        appreciation_threshold = 0.40
        cond_appreciation = False
        if not np.isnan(growth_pct):
            cond_appreciation = growth_pct >= appreciation_threshold

        # 2. Price per SqFt vs city median
        city_ppsf_median = (
            df[df["City"] == city]["Price_per_SqFt"].median()
            if not df[df["City"] == city].empty
            else df["Price_per_SqFt"].median()
        )
        cond_ppsf = price_per_sqft <= city_ppsf_median

        # 3. BHK >= 2
        cond_bhk = bhk >= 2

        # 4. Public transport accessibility >= dataset median
        pta_num_input = pta_mapping.get(public_transport, 2)
        cond_transport = pta_num_input >= median_pta_num_global

        # 5. Parking available
        parking_num_input = 1 if str(parking).lower() == "yes" else 0
        cond_parking = parking_num_input > 0

        # Investment score
        conditions = [
            ("High Appreciation (≥ 40% in 5 yrs)", cond_appreciation),
            ("Price per SqFt below city median", cond_ppsf),
            ("Sufficient BHK (≥ 2)", cond_bhk),
            ("Good Public Transport (≥ median)", cond_transport),
            ("Parking Space Available", cond_parking),
        ]

        score = sum(1 for _, c in conditions if c)
        st.write(f"**Investment Score (Rule-based): `{score} / 5`**")

        # Colour hint
        if score >= 4:
            st.success("Excellent score based on rules (likely a strong long-term investment).")
        elif score == 3:
            st.info("Decent score based on rules (moderate investment potential).")
        else:
            st.warning("Low rule-based score. Model and rules both are cautious about this investment.")

        # Table-style breakdown
        breakdown_rows = []
        for name, cond in conditions:
            status = "✅ Meets criteria" if cond else "❌ Does not meet"
            breakdown_rows.append({"Criteria": name, "Status": status})

        st.table(pd.DataFrame(breakdown_rows))

        st.caption(
            "Note: This rule-based score was used to generate the `Good_Investment` labels "
            "for training the classifier. The ML model may sometimes disagree slightly with "
            "these raw rules due to learning patterns from the full dataset."
        )

# ---- TAB 2: EDA ----
with tab_eda:
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    eda1, eda2, eda3, eda4 = st.tabs(
        ["Price & Size", "Location-based", "Relationships", "Investment & Amenities"]
    )

    # 1–5 Price & Size
    with eda1:
        colA, colB = st.columns(2)
        with colA:
            st.write("**1. Distribution of property prices**")
            st.pyplot(plot_price_distribution(df))

        with colB:
            st.write("**2. Distribution of property sizes**")
            st.pyplot(plot_size_distribution(df))

        st.write("**3 & 4. Size vs Price**")
        st.pyplot(plot_price_vs_size(df))

        st.write("**5. Outliers in Price per SqFt**")
        fig, ax = plt.subplots()
        ax.boxplot(df["Price_per_SqFt"].dropna())
        ax.set_ylabel("Price per SqFt")
        st.pyplot(fig)

    # 6–10 Location-based
    with eda2:
        colC, colD = st.columns(2)
        with colC:
            st.write("**6. Average price per SqFt by State**")
            state_ppsf = (
                df.groupby("State")["Price_per_SqFt"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(state_ppsf)

        with colD:
            st.write("**7. Average property price by City**")
            city_price = (
                df.groupby("City")["Price_in_Lakhs"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(city_price)

        st.write("**8. Median age of properties by Locality**")
        locality_age = (
            df.groupby("Locality")["Age_of_Property"].median().sort_values()
        )
        st.bar_chart(locality_age.head(30))

        colE, colF = st.columns(2)
        with colE:
            st.write("**9. BHK distribution across cities (mean BHK)**")
            bhk_dist = (
                df.pivot_table(index="City", values="BHK", aggfunc="mean")
                .sort_values("BHK", ascending=False)
            )
            st.bar_chart(bhk_dist)

        with colF:
            st.write("**10. Price trends for top 5 most expensive localities**")
            loc_price = (
                df.groupby("Locality")["Price_in_Lakhs"]
                .median()
                .sort_values(ascending=False)
                .head(5)
            )
            st.bar_chart(loc_price)

    # 11–15 Correlations & relationships
    with eda3:
        st.write("**11. Correlation between numeric features**")
        num_cols = df.select_dtypes(include=np.number).columns
        corr = df[num_cols].corr()
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

        colG, colH = st.columns(2)
        with colG:
            st.write("**12. Nearby schools vs Price per SqFt**")
            schools_ppsf = df.groupby("Nearby_Schools")["Price_per_SqFt"].mean()
            st.line_chart(schools_ppsf)

        with colH:
            st.write("**13. Nearby hospitals vs Price per SqFt**")
            hosp_ppsf = df.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean()
            st.line_chart(hosp_ppsf)

        colI, colJ = st.columns(2)
        with colI:
            st.write("**14. Price vs Furnished Status**")
            furn_price = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean()
            st.bar_chart(furn_price)

        with colJ:
            st.write("**15. Price per SqFt vs Facing direction**")
            face_ppsf = df.groupby("Facing")["Price_per_SqFt"].mean()
            st.bar_chart(face_ppsf)

    # 16–20 Investment / Amenities / Ownership
    with eda4:
        colK, colL = st.columns(2)
        with colK:
            st.write("**16. Properties by Owner Type**")
            owner_counts = df["Owner_Type"].value_counts()
            st.bar_chart(owner_counts)

        with colL:
            st.write("**17. Properties by Availability Status**")
            avail_counts = df["Availability_Status"].value_counts()
            st.bar_chart(avail_counts)

        colM, colN = st.columns(2)
        with colM:
            st.write("**18. Effect of Parking Space on Price**")
            park_price = df.groupby("Parking_Space")["Price_in_Lakhs"].mean()
            st.bar_chart(park_price)

        with colN:
            st.write("**19. Amenities vs Price per SqFt**")
            amen_ppsf = df.groupby("Amenities")["Price_per_SqFt"].mean()
            st.bar_chart(amen_ppsf)

        st.write("**20. Public Transport Accessibility vs Price per SqFt**")
        st.pyplot(plot_public_transport_vs_ppsf(df))

# ---- TAB 3: Feature importance ----
with tab_feat:
    st.subheader("📈 Feature Importance (Explainability)")

    colFI1, colFI2 = st.columns(2)

    with colFI1:
        st.write("### Classifier Feature Importance")
        try:
            fi_clf = get_tree_feature_importance(clf, top_n=20)
            st.dataframe(fi_clf)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.barh(fi_clf["feature"][::-1], fi_clf["importance"][::-1])
            ax.set_title("Top Features - Classifier")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not compute classifier feature importance: {e}")

    with colFI2:
        st.write("### Regressor Feature Importance")
        try:
            fi_reg = get_tree_feature_importance(reg, top_n=20)
            st.dataframe(fi_reg)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.barh(fi_reg["feature"][::-1], fi_reg["importance"][::-1])
            ax.set_title("Top Features - Regressor")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not compute regressor feature importance: {e}")

st.markdown("---")
st.caption(
    "Models trained on Indian housing dataset. "
    "Classification target: Good_Investment, Regression target: Future_Price_5Y."
)
