import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import random

# Sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

# Simulate User Interaction Data
def generate_user_data(n=10):
    data = {
        "click_count": np.random.rand(n),
        "hesitation_time": np.random.rand(n),
    }
    df = pd.DataFrame(data)
    df["bias_detected"] = np.where(df["click_count"] > 0.7, "Recency Bias", "None")
    df["ui_modification"] = df["bias_detected"].apply(
        lambda x: "Inject Older Content in Feed" if x != "None" else "No Change"
    )
    return df

# Exposure Diversity Score Calculation
def calculate_ed_score(data):
    unique_mods = data["ui_modification"].nunique()
    return round(unique_mods / len(data["ui_modification"].unique()), 3)

# Main UI
st.set_page_config(page_title="Ethically Adaptive UI Prototype", layout="wide")
st.title("🧠 Ethically Adaptive UI Prototype")

# Input area for user sentiment
st.subheader("🔤 User Text Sentiment")
user_input = st.text_area("Type any user message here for sentiment analysis:")

if user_input:
    result = sentiment_pipeline(user_input)
    st.success(f"**{result[0]['label']}** sentiment with a score of **{result[0]['score']:.3f}**")

# Generate and show user data simulation
st.subheader("📊 Simulated User Interaction Data")
df = generate_user_data(10)
st.dataframe(df)

# Sentiment on pre-written texts
st.subheader("🧠 Sentiment Analysis of Sample Texts")
sample_texts = ["I love this app!", "This is terrible.", "I hate this.", "Worst experience ever."]
results = sentiment_pipeline(sample_texts)
sentiment_df = pd.DataFrame(results)
st.dataframe(sentiment_df)

# Diversity score
ed_score = calculate_ed_score(df)
st.subheader("📏 Exposure Diversity Score")
st.metric("Diversity Score", ed_score)
