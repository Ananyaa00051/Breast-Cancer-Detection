
# 🩺 Breast Cancer Prediction and Explainability App

A powerful and interactive **machine learning web application** built using **Streamlit**, powered by **XGBoost**, **SHAP Explainability**, and enhanced by **LLM-backed chat integration** for intuitive understanding of model predictions.

> 🔍 Predict. 📊 Understand. 💬 Chat.

---

## 🚀 What This App Does

✅ Predicts **breast cancer** (Malignant or Benign) using clinical features  
✅ Visualizes **feature importance** using SHAP values  
✅ Allows users to **chat with an AI assistant** to understand the prediction  
✅ Offers detailed **feature descriptions** for transparency  
✅ Integrates **Mistral LLM via Ollama** to power explanations  

---

## 🧠 Technologies Used

| Component        | Description                                      |
|------------------|--------------------------------------------------|
| `XGBoost`        | High-performance ML model for binary classification |
| `SHAP`           | Explains model predictions with shapley values     |
| `Ollama + Mistral` | Local LLM integration for natural language explanation |
| `Streamlit`      | Framework for building the UI                      |
| `Pandas, NumPy`  | Data processing and numerical computation         |
| `scikit-learn`   | Dataset and ML utilities                          |

---

## 📁 Project Structure

```

📦 breast-cancer-prediction-app/
├── app.py                # Streamlit frontend
├── model.pkl             # Trained XGBoost model
├── chatbot.py            # Kash Chat – LLM-assisted chatbot logic
├── feature\_description.py# Feature info with medical context
├── ollama\_helper.py      # Ollama integration using Mistral
├── shap\_viz.py           # SHAP visualization utilities
├── requirements.txt      # Python dependencies
└── README.md

````

---

## 🎯 Core Features

### 🧬 Breast Cancer Prediction

- Uses the `XGBoostClassifier` trained on the **Wisconsin Breast Cancer Dataset**
- Achieves ~97% accuracy
- Classifies input as **Malignant** or **Benign**

---

### 📈 SHAP-Based Explainability

- Visual SHAP plot for feature impact
- Clear insight into **why** a certain prediction was made
- Helps clinicians understand key contributing factors

---

### 💬 Chat with AI – *Kash Chat*

A custom-built chatbot that uses **Mistral** via **Ollama** locally to:

- Explain what each feature means
- Describe why a certain output was given
- Help users interpret SHAP visuals
- Act like a “tech + healthcare” assistant

---

### 🔍 Feature Descriptions

Get instant clarity on every medical term and parameter, including:

- `mean radius`
- `texture`
- `concavity`
- `symmetry`
- ...and more.

Each feature is documented in layman-friendly terms with real-world analogies.

---

## 🛠 How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/breast-cancer-predictor
cd breast-cancer-predictor
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

Ensure Ollama is installed and Mistral model is pulled:

```bash
ollama run mistral
```

### 3. Launch the App

```bash
streamlit run app.py
```

---


## 📦 Requirements Summary

* Python ≥ 3.9
* streamlit
* xgboost
* pandas
* numpy
* shap
* scikit-learn
* ollama
* mistral (model for Ollama)

---

## 🔒 Disclaimer

> This tool is intended for **educational and prototyping purposes only**. It is not a replacement for medical advice or diagnosis.

---


