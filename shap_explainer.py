import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Load model and scaler
model = joblib.load("xgb_top10_model.pkl")
scaler = joblib.load("scaler_top10.pkl")

# Feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_worst'
]

def explain_prediction(input_data):
    input_df = pd.DataFrame(input_data, columns=feature_names)
    scaled_input = scaler.transform(input_df)

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)

    if isinstance(shap_values, list):
        shap_val = shap_values[1][0]
        base_val = explainer.expected_value[1]
    else:
        shap_val = shap_values[0]
        base_val = explainer.expected_value

    # 🌊 Waterfall Plot
    st.subheader("🔍 SHAP Feature Contribution")
    st.markdown("This plot shows how each feature contributed to the diagnosis.")

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_val,
        base_values=base_val,
        data=scaled_input[0],
        feature_names=feature_names
    ))
    st.pyplot(fig)

    # 🧠 NATURAL LANGUAGE SUMMARY
    st.subheader("📘 AI-Powered Diagnosis Explanation")

    # Get top 3 features by absolute SHAP impact
    abs_vals = np.abs(shap_val)
    top_idx = abs_vals.argsort()[::-1][:3]  # Top 3
    explanation = ""

    for idx in top_idx:
        name = feature_names[idx]
        val = input_data[0][idx]
        impact = shap_val[idx]
        direction = "increased" if impact > 0 else "decreased"
        contribution = f"- The value of **{name}** was `{val:.2f}`, which **{direction}** the likelihood of cancer.\n"
        explanation += contribution

    st.markdown(f"""
Based on the model's interpretation:

- The value of **{name}** was `{val:.2f}`, which **{direction}** the likelihood of cancer.
...
""")
