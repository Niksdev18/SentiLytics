# dashboard/app.py

import streamlit as st
import pandas as pd
import os, sys
from datetime import datetime
import plotly.express as px
from PIL import Image

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.predict_realtime import predict_sentiment

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="SentiLytics - Financial Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Branding (Logo + Tagline)
# ---------------------------
logo_path = "assets/sentilytics_logo.png"  # put your logo inside dashboard/assets
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=180)

st.markdown(
    "<h1 style='color:#2E8B57;'>SentiLytics</h1>", unsafe_allow_html=True
)
st.markdown("*Turning Financial News into Market Insights 🚀*")

# ---------------------------
# Sidebar: Login
# ---------------------------
st.sidebar.title("🔑 Login / Signup")
auth_option = st.sidebar.radio("Choose", ["Login", "Sign in with Google"])

if auth_option == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_btn = st.sidebar.button("Login")
    if not (username == "admin" and password == "admin" and login_btn):
        if login_btn:
            st.sidebar.error("Invalid credentials ❌")
        st.stop()
    else:
        st.sidebar.success("Logged in ✅")
else:
    st.sidebar.info("👉 Google Sign-In coming soon...")

# ---------------------------
# Navigation Tabs
# ---------------------------
tabs = st.tabs(["🏠 Home", "📰 Sentiment Prediction", "📈 Market Trends", "🗞️ News Insights"])

# ---------------------------
# Home Tab
# ---------------------------
with tabs[0]:
    st.header("📢 Welcome to SentiLytics")
    st.markdown(
        """
        This dashboard helps you analyze **financial news sentiment** 
        and its impact on **stock market movements** in real time.  
        
        🔹 Enter a news headline → Get sentiment prediction  
        🔹 Compare sentiment with stock price trends  
        🔹 Explore top financial headlines & their impact  
        """
    )

    st.markdown("---")
    st.markdown("### Why SentiLytics?")
    col1, col2, col3 = st.columns(3)
    col1.metric("Models", "Transformer (FinBERT)")
    col2.metric("Accuracy", "85%")
    col3.metric("Use Case", "Finance/Trading")

# ---------------------------
# Sentiment Prediction Tab
# ---------------------------
with tabs[1]:
    st.header("📰 Predict Sentiment of Financial News")
    news_input = st.text_area("Enter headline(s), one per line:")

    if st.button("🔍 Predict Sentiment"):
        if news_input.strip():
            headlines = [h.strip() for h in news_input.strip().split("\n") if h.strip()]
            predictions = predict_sentiment(headlines)
            df_results = pd.DataFrame({"Headline": headlines, "Predicted Sentiment": predictions})
            st.dataframe(df_results, use_container_width=True)

            # Dummy market trend prediction (placeholder)
            st.subheader("📊 Market Impact (Prediction)")
            impact = ["Increase 📈" if p == "Positive" else "Decrease 📉" if p == "Negative" else "Neutral ➖" for p in predictions]
            df_results["Impact"] = impact
            st.table(df_results[["Headline", "Predicted Sentiment", "Impact"]])
        else:
            st.warning("Please enter at least one headline.")

# ---------------------------
# Market Trends Tab
# ---------------------------
with tabs[2]:
    st.header("📈 Historical Sentiment vs Stock Price")
    data_file = "data/processed/realtime/market_sentiment_20250930.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        required = ["published", "avg_sentiment", "Close"]
        if all(col in df.columns for col in required):
            fig = px.line(
                df, x="published", y=["avg_sentiment", "Close"],
                title="Sentiment vs Stock Price", labels={"value": "Score / Price"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Required columns {required} not found in {data_file}")
    else:
        st.info("Run market_sentiment.py to generate merged CSV.")

# ---------------------------
# News Insights Tab
# ---------------------------
with tabs[3]:
    st.header("🗞️ Latest 30 Financial News Sentiment")
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        if not df.empty:
            latest_30 = df.tail(30).reset_index(drop=True)
            for idx, row in latest_30.iterrows():
                with st.expander(f"{row['published']} | Headline {idx+1}"):
                    st.markdown(f"**Sentiment Score:** {row['avg_sentiment']:.2f}")
                    trend_slice = df.iloc[max(0, idx-10):idx+1][["avg_sentiment", "Close"]]
                    fig = px.line(trend_slice, y=["avg_sentiment", "Close"], 
                                  title="Local Trend", labels={"value": "Score/Price"})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available.")
    else:
        st.warning("No CSV file found.")
