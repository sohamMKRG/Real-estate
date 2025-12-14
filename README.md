# 🏠 Real Estate Investment Advisor

**Predicting Property Profitability & Future Value using Machine Learning**

---

## 📌 Project Overview

The **Real Estate Investment Advisor** is a machine learning–based decision support system designed to help real estate investors and buyers evaluate properties based on **long-term investment potential**.

The application:

* **Classifies** whether a property is a *Good Investment* or *Not Recommended*
* **Predicts** the estimated property price after **5 years**
* Provides **rule-based explainability** so users understand *why* a property is accepted or rejected
* Includes **EDA insights** to explore market trends

The project is deployed as an interactive **Streamlit web application**.

---

## 🎯 Problem Statement

Real estate investors often face difficulty in assessing:

* Whether a property is fairly priced
* Its long-term appreciation potential
* The impact of amenities, transport, and location on future value

This project solves that by using **machine learning + domain rules** to provide data-driven investment recommendations.

---

## 💼 Business Use Cases

* 📈 Assist investors in identifying high-return properties
* 🏙 Support buyers in choosing undervalued properties in growing areas
* 🤖 Automate property investment analysis for real estate platforms
* 🔍 Improve transparency and trust using explainable AI

---

## 🧠 Solution Approach

### 1️⃣ Data Preprocessing

* Handled missing values and duplicates
* Feature engineering:

  * Price per SqFt
  * Property age
  * Rule-based investment score
* Encoded categorical variables
* Scaled numerical features

---

### 2️⃣ Exploratory Data Analysis (EDA)

Answered 20+ business questions including:

* Price and size distributions
* Location-wise price trends
* Impact of transport, parking, and amenities
* Correlation between features and price

---

### 3️⃣ Machine Learning Models

#### 🔹 Classification

* **Target:** `Good_Investment` (Yes / No)
* **Model:** Random Forest Classifier
* **Metrics:** Accuracy, F1-score

#### 🔹 Regression

* **Target:** `Future_Price_5Y`
* **Model:** Random Forest Regressor
* **Metrics:** RMSE, MAE, R²

---

### 4️⃣ Explainability

A **rule-based investment score (X / 5)** is shown in the app:

* High appreciation (≥ 40%)
* Price per SqFt below city median
* BHK ≥ 2
* Good public transport
* Parking availability

This helps users understand *why* a property is recommended or rejected.

---

## 🖥️ Streamlit Application Features

* User-friendly property input form
* Investment recommendation (Good / Not Recommended)
* Future price prediction (5 years)
* Expected appreciation & profit
* Model confidence score
* Rule-based investment score breakdown
* Interactive EDA visualizations
* Feature importance charts

---

## 🗂️ Project Structure

```
real_estate_investment_advisor/
│
├── app.py                       # Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── data/
│   └── processed_housing_small.csv   # Lightweight dataset for deployment
│
├── models/
│   ├── classifier.pkl
│   └── regressor.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train_models.py
│   ├── make_small_dataset.py
│   └── utils.py
│
└── .gitignore
```

---

## 📊 Dataset Information

* **Original Dataset:** Indian Housing Prices
* **Note:**
  Large raw datasets are excluded due to GitHub size limits.
  A **processed lightweight dataset** is included for deployment and inference.

---

## ⚙️ Installation & Local Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/sohamMKRG/Real-estate.git
cd Real-estate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🚀 Deployment

The application is **ready for Streamlit Cloud deployment** using this repository.

Steps:

1. Connect GitHub repo to Streamlit Cloud
2. Select `app.py` as entry point
3. Deploy

---

## 📈 Model Performance (Summary)

* **Classifier**

  * Accuracy: ~94%
  * F1-Score: ~96%

* **Regressor**

  * RMSE: Low
  * R²: ~1.0 (synthetic / engineered target)

---

## 🧪 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Joblib
* Git & GitHub

---

## 📌 Future Improvements

* Integrate real historical price appreciation data
* Add rental yield prediction
* City-specific growth models
* Database integration for large-scale deployment
* Advanced explainability (SHAP values)

---

## 👤 Author

**Soham Mukherjee**
B.Tech Undergraduate
Domain: Machine Learning | Data Analytics | Real Estate Analytics

---

## ⭐ Acknowledgements

* Indian Housing Dataset
* Streamlit & Scikit-learn community


