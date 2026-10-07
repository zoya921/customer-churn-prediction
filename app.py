import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a telecom customer is likely to churn.")

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = df.drop("customerID", axis=1)

# Feature engineering
df["AverageMonthlySpend"] = (
    df["TotalCharges"] / df["tenure"].replace(0, np.nan)
)

df["AverageMonthlySpend"] = df["AverageMonthlySpend"].fillna(
    df["MonthlyCharges"]
)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True, dtype=int)

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# User inputs
st.subheader("Enter Customer Information")

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=monthly_charges * max(tenure, 1)
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

# Prediction
if st.button("Predict Churn"):

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=X.columns
    )

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    average_spend = (
        total_charges / tenure if tenure > 0
        else monthly_charges
    )

    input_data["AverageMonthlySpend"] = average_spend

    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    if internet_service == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet_service == "No":
        input_data["InternetService_No"] = 1

    if paperless_billing == "Yes":
        input_data["PaperlessBilling_Yes"] = 1

    # Long-term contract feature
    input_data["LongTermContract"] = (
        input_data["Contract_One year"] |
        input_data["Contract_Two year"]
    ).astype(int)

    # Make sure column order matches training data
    input_data = input_data[X.columns]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ Customer is likely to churn. "
            f"Churn probability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Customer is likely to stay. "
            f"Churn probability: {probability:.2%}"
        )
