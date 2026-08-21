import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="FinSight AI", page_icon="💰", layout="wide")

st.title("💰 FinSight AI - Fraud Detection Dashboard")
st.markdown("AI-powered cash flow & expense anomaly detection")

@st.cache_data
def load_data():
    df = pd.read_csv('data/transactions_sample.csv')
    return df

df = load_data()

st.header("📊 Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", len(df))

with col2:
    fraud_count = df['is_fraud'].sum()
    st.metric("Fraud Cases Detected", fraud_count)

with col3:
    fraud_rate = (fraud_count / len(df)) * 100
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

st.header("📋 Sample Transactions")
st.dataframe(df.head(20))

import matplotlib.pyplot as plt
import seaborn as sns

st.header("🔍 Fraud Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Amount: Genuine vs Fraud")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.boxplot(x='is_fraud', y='amt', data=df, showfliers=False, ax=ax1)
    ax1.set_xlabel('Is Fraud (0 = Genuine, 1 = Fraud)')
    ax1.set_ylabel('Amount ($)')
    st.pyplot(fig1)

with col2:
    st.subheader("Fraud Rate (%) by Category")
    fraud_by_category = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False) * 100
    fig2, ax2 = plt.subplots(figsize=(6,4))
    fraud_by_category.plot(kind='bar', color='indianred', ax=ax2)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Fraud Rate (%)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)

st.subheader("Fraud Rate (%) by Hour of Day")
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
fraud_by_hour = df.groupby('hour')['is_fraud'].mean() * 100

fig3, ax3 = plt.subplots(figsize=(12,4))
fraud_by_hour.plot(kind='line', marker='o', color='darkred', ax=ax3)
ax3.set_xlabel('Hour (24-hour format)')
ax3.set_ylabel('Fraud Rate (%)')
ax3.grid(True)
st.pyplot(fig3)

st.header("🎯 Test a Transaction")
st.markdown("Enter transaction details to check if it's likely fraud")

model = joblib.load('models/fraud_model.pkl')
le_category = joblib.load('models/le_category.pkl')
le_gender = joblib.load('models/le_gender.pkl')

col1, col2, col3 = st.columns(3)

with col1:
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
    category = st.selectbox("Category", le_category.classes_)

with col2:
    gender = st.selectbox("Gender", le_gender.classes_)
    hour = st.slider("Hour of Transaction (24-hr)", 0, 23, 12)

with col3:
    city_pop = st.number_input("City Population", min_value=0, value=50000, step=1000)
    lat = st.number_input("Latitude", value=40.0, format="%.4f")

long = st.number_input("Longitude", value=-75.0, format="%.4f")

if st.button("🔍 Check for Fraud", type="primary"):
    category_encoded = le_category.transform([category])[0]
    gender_encoded = le_gender.transform([gender])[0]
    
    input_data = pd.DataFrame({
        'amt': [amt],
        'category_encoded': [category_encoded],
        'gender_encoded': [gender_encoded],
        'hour': [hour],
        'city_pop': [city_pop],
        'lat': [lat],
        'long': [long],
        'merch_lat': [lat],
        'merch_long': [long]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ FRAUD ALERT! This transaction has a {probability*100:.1f}% probability of being fraudulent.")
    else:
        st.success(f"✅ This transaction looks genuine. Fraud probability: {probability*100:.1f}%")