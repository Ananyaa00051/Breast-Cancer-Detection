import streamlit as st
import numpy as np
import joblib
from shap_explainer import explain_prediction
from llm_explainer import explain_medical_term
from kash_chat import chat_with_kash  # ✅ Move this import to top

# ✅ Load trained model and scaler
model = joblib.load("xgb_top10_model.pkl")
scaler = joblib.load("scaler_top10.pkl")

# ✅ Custom background and styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/premium-photo/pink-ribbon-stethoscope-colored-background-breast-cancer-awareness-month-women-s-health-care-concept-symbol-fight-against-oncology-copy-space_183793-3289.jpg');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        color: #ffffff;
    }
    header, footer {visibility: hidden;}
    .block-container {
        backdrop-filter: blur(8px);
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 15px;
        padding: 2rem;
        margin-top: -4rem;
    }
    h1, h2, h3, h4 {
        color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ App container
st.markdown("<div class='block-container'>", unsafe_allow_html=True)
st.title("🔬 Breast Cancer Diagnosis Tool")
st.markdown("This AI tool predicts if a tumor is **Malignant (cancerous)** or **Benign** based on key clinical features.")

st.subheader("🧪 Input Clinical Features:")
# 🔢 Input fields for top 10 features
radius_mean = st.number_input("Radius Mean", min_value=0.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0)
area_mean = st.number_input("Area Mean", min_value=0.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0)
compactness_mean = st.number_input("Compactness Mean", min_value=0.0)
concavity_mean = st.number_input("Concavity Mean", min_value=0.0)
symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0)
radius_worst = st.number_input("Radius Worst", min_value=0.0)

# 📘 LLM Feature Explainer (Independent)
st.subheader("📘 Learn What a Medical Feature Means")
term = st.selectbox("Select a feature to understand:", [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "symmetry_mean", "fractal_dimension_mean", "radius_worst"
])
if st.button("📖 Explain This Term"):
    with st.spinner("Thinking..."):
        explanation = explain_medical_term(term)
    st.success(explanation)

# 🧬 Predict Diagnosis Button
input_data = None
if st.button("🧬 Predict Diagnosis"):
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                            smoothness_mean, compactness_mean, concavity_mean,
                            symmetry_mean, fractal_dimension_mean, radius_worst]])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][prediction]

    if prediction == 1:
        st.markdown("<h2 style='color:#ff4c4c;'>🛑 Diagnosis: Malignant (Cancerous)</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:#33cc33;'>✅ Diagnosis: Benign (Non-cancerous)</h2>", unsafe_allow_html=True)

    st.markdown(f"**Confidence**: {round(probability * 100, 2)}%")
    st.warning("⚠️ This tool is for educational/demo purposes only. Please consult a medical professional for real diagnosis.")

    # 📊 SHAP Explanation
    st.subheader("📊 SHAP Feature Contribution")
    explain_prediction(input_data)

    # 🧠 LLM-generated plain English explanations
    st.subheader("🧠 Key Features That Influenced This Result:")
    for feat in ["radius_worst", "concavity_mean", "texture_mean"]:
        with st.spinner(f"Explaining '{feat}'..."):
            exp = explain_medical_term(feat)
        st.markdown(f"**{feat.replace('_', ' ').title()}**: {exp}")

# 💬 Chatbot - KASH
st.subheader("💬 Talk to KASH - Your Breast Cancer Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask KASH anything about breast cancer:")

if user_input:
    with st.spinner("KASH is thinking..."):
        response = chat_with_kash(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append(f"User: {user_input}")
        st.session_state.chat_history.append(f"KASH: {response}")
        st.markdown(f"**KASH:** {response}")

# ℹ️ Note
st.markdown("</div>", unsafe_allow_html=True)
st.info("ℹ️ This tool is created for awareness and learning. Not a substitute for professional medical advice.")
