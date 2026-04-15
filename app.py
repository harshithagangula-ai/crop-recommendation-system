import streamlit as st
import numpy as np
import pickle
import json
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Recommendation System 🌾",
    page_icon="🌱",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("model.pkl", "rb"))
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_assets()

# ---------------- STYLES ----------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1b5e20;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="main-header">🌾 Smart Crop Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Leverage data-driven insights to choose the most suitable crop for your region</p>', unsafe_allow_html=True)

# ---------------- SIDEBAR INFO ----------------
with st.sidebar:
    st.image("feature_importance.png", caption="Feature Importance", use_container_width=True)
    st.markdown("### 📊 Model Overview")
    st.metric("Accuracy", f"{metadata['accuracy']*100:.1f}%")
    st.metric("Cross Validation", f"{metadata['cv_accuracy']*100:.1f}%")
    st.markdown("---")
    
    feature_df = pd.DataFrame({
        "Feature": list(metadata["feature_importance"].keys()),
        "Importance": list(metadata["feature_importance"].values())
    }).sort_values("Importance", ascending=False)

    st.markdown("### 🧠 Top Features")
    st.dataframe(feature_df.head(5), use_container_width=True, hide_index=True)
    st.markdown("---")
    st.info("""
    - Keep soil pH between **6–7** for most crops  
    - Ensure adequate **Nitrogen and Potassium** levels  
    - Use **local weather averages** for better accuracy  
    """)

# ---------------- USER INPUT ----------------
st.markdown("### 🌍 Input Soil and Weather Conditions")

c1, c2, c3, c4 = st.columns(4)

with c1:
    n = st.number_input("Nitrogen (N)", 0, 150, 40)
with c2:
    p = st.number_input("Phosphorus (P)", 0, 150, 50)
with c3:
    k = st.number_input("Potassium (K)", 0, 150, 50)
with c4:
    ph = st.number_input("pH Value", 0.0, 14.0, 6.5, 0.1)

c5, c6, c7 = st.columns(3)
with c5:
    temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
with c6:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, 1.0)
with c7:
    rainfall = st.number_input("Rainfall (mm/year)", 0.0, 300.0, 100.0, 1.0)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🌱 Get Recommendation", use_container_width=True):
    input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_crops = [(model.classes_[i], probabilities[i]) for i in top_indices]

    # Result Display
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);
                padding:2rem;border-radius:15px;text-align:center;border:2px solid #81c784;">
        <h3 style="color:#1b5e20;">🌾 Recommended Crop:</h3>
        <h1 style="color:#1b5e20;font-weight:800;">{prediction.title()}</h1>
        <p style="font-size:1.1rem;color:#2e7d32;">Confidence: {top_crops[0][1]*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Probability Graph
    prob_df = pd.DataFrame({
        "Crop": [c.title() for c, _ in top_crops],
        "Probability": [p*100 for _, p in top_crops]
    })
    fig = px.bar(prob_df, x="Crop", y="Probability", color="Crop",
                 color_discrete_sequence=px.colors.sequential.Greens,
                 title="Top Crop Predictions (% Confidence)")
    st.plotly_chart(fig, use_container_width=True)

    # Input Summary
    with st.expander("📋 Input Summary"):
        st.dataframe(pd.DataFrame({
            "Parameter": ["Nitrogen", "Phosphorus", "Potassium", "pH", "Temperature", "Humidity", "Rainfall"],
            "Value": [n, p, k, ph, temp, humidity, rainfall]
        }), hide_index=True)

    # Detailed Recommendations
    st.markdown("### 🌱 Alternative Crop Matches")
    alt_cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (crop, prob) in enumerate(top_crops):
        with alt_cols[i]:
            st.metric(f"{medals[i]} {crop.title()}", f"{prob*100:.1f}%")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#777;font-size:0.9rem;'>"
    "Built with ❤️ using Streamlit & Machine Learning — © 2026"
    "</p>",
    unsafe_allow_html=True
)
