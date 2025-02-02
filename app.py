import streamlit as st
import joblib
import numpy as np

# Load trained fraud detection model & imputer
model = joblib.load("fraud_detection_model.pkl")
imputer = joblib.load("feature_imputer.pkl")  # Load pre-trained imputer

# 🎨 **Custom Page Styling**
st.set_page_config(page_title="Fraud Detection App", page_icon="🚀", layout="wide")

# CSS for better design
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stButton button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 24px;
        }
        .stButton button:hover {
            background-color: #ff2222;
        }
        .stTextInput, .stNumberInput {
            border-radius: 10px;
            border: 1px solid #ccc;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🏠 **Header Section**
st.title("🔍 Credit Card Fraud Detection App")
st.write("💳 **Predict whether a transaction is fraudulent based on key details.**")

# 📂 **Organizing Input Section**
st.sidebar.header("Enter Transaction Details:")
amount = st.sidebar.number_input("💰 Transaction Amount ($)", min_value=0.0, format="%.2f")
time = st.sidebar.number_input("⏳ Transaction Time (Seconds)", min_value=0.0, format="%.2f")
spending_behavior = st.sidebar.number_input("📊 Customer Spending Behavior Score")
risk_level = st.sidebar.number_input("⚠️ Transaction Risk Level")
account_trust = st.sidebar.number_input("🔐 Account Trust Score")
merchant_reputation = st.sidebar.number_input("🏪 Merchant Reputation Score")

# Create an array with 30 features (set 24 as NaN)
partial_input = np.array([[amount, time, spending_behavior, risk_level, account_trust, merchant_reputation] + [np.nan] * 24])  # Missing 24 features

# Fill in missing features using KNNImputer
complete_input = imputer.transform(partial_input)  # Now we have 30 features

# 🚀 **Prediction Button**
if st.sidebar.button("🔎 Check for Fraud"):
    prediction = model.predict(complete_input)[0]

    # 🎯 **Display Results with Animation**
    st.markdown("## **🔍 Fraud Detection Result**")
    with st.spinner("Processing..."):
        st.write("✅ **Analyzing transaction details...**")
    
    # 🚨 Fraud detected
    if prediction == 1:
        st.error("🚨 **Fraudulent Transaction Detected!** ⚠️")
        st.warning("🔴 Immediate Action Recommended.")
    else:
        st.success("✅ **Transaction is Legitimate.** No fraud detected.")

# 📌 **Footer**
st.markdown("---")
st.markdown("🔐 **Secure AI-based Fraud Detection** | 🚀 Developed with Streamlit & Machine Learning")
