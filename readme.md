
# 🩺 Breast Cancer Prediction & Local AI Assistant - KASH

An interactive Streamlit-based web application that enables users to:

- Predict the likelihood of breast cancer using an XGBoost model
- Understand predictions using SHAP explainability
- Chat with a locally-hosted AI assistant named **KASH** (Knowledge Assistant for Self-Health) powered by **Ollama + Mistral LLM**



## 🚀 Features

### 1. 🎯 **Breast Cancer Prediction using XGBoost**
- Trained on the Wisconsin Breast Cancer dataset
- Uses features like `mean radius`, `mean texture`, etc.
- Outputs binary classification: **Benign** or **Malignant**
- Scikit-learn preprocessing and model pipeline

### 2. 📊 **Feature Importance Visualization**
- Built-in SHAP (SHapley Additive exPlanations) plots
- Understand how each feature impacts the model's prediction

### 3. 🧠 **KASH: Local AI Chatbot Assistant**
- Chatbot interface within the app
- Uses the `mistral` model from **Ollama**
- Runs fully offline (no API needed)
- Supports custom prompts like:
  - “What is breast cancer?”
  - “How to interpret biopsy results?”
  - “Explain the role of radius_mean in diagnosis.”

### 4. 🗃️ **Explainability**
- SHAP summary plots provide transparent decision-making insights



## 💻 Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core programming |
| **XGBoost** | Machine learning classifier |
| **SHAP** | Model explainability |
| **Streamlit** | Web UI |
| **Ollama** | LLM runtime |
| **Mistral** | Local large language model |
| **Matplotlib & Seaborn** | Visualization |

---

## 🧰 Setup Instructions

1. **Clone the repository**:

   git clone https://github.com/your-username/breast-cancer-kash.git
   cd breast-cancer-kash


2. **Install dependencies**:

   
   pip install -r requirements.txt
  

3. **Run Ollama with Mistral**:

   ollama run mistral
  

4. **Launch the app**:


   streamlit run app.py
  

---

## 🧪 Sample Prompts for KASH

```
- What are the early symptoms of breast cancer?
- How accurate is a biopsy?
- Can you explain the SHAP plot I see?
- What is a benign tumor?
- Is radius_mean a strong indicator of cancer?
```

---

## 📂 File Structure

```
📁 breast-cancer-kash/
├── app.py               # Main Streamlit app
├── kash_chat.py         # Local LLM chatbot logic
├── model.pkl            # Trained XGBoost model
├── shap_plot.png        # SHAP summary image (optional)
├── requirements.txt     # Required Python packages
└── README.md            # This file
```

---

## 🌐 Deploy on Streamlit Community Cloud

> ❗ Make sure the repository is public on GitHub.

1. Push the project to a GitHub repository
2. Go to: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Select your repo and deploy

---

## 🧠 Credits

Developed by: Ananyaa Sharrma
LLM Integration via: [Ollama](https://ollama.com)
Dataset: [Wisconsin Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

---

## 🛡️ Disclaimer

This project is intended for **educational purposes only**. It is not a replacement for professional medical diagnosis or advice.

