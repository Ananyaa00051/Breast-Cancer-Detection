
# 🩺 Breast Cancer Prediction App

A simple yet powerful machine learning web application built with **Streamlit** that predicts the likelihood of breast cancer using input medical parameters. This project uses the **XGBoost Classifier** trained on the Breast Cancer Wisconsin dataset.


## 🚀 Features

- 🔮 Predicts whether a breast tumor is **Malignant** or **Benign**
- 📊 Clean and intuitive UI built with Streamlit
- 📁 Lightweight and beginner-friendly project structure
- 🧠 Powered by **XGBoost**, a high-performance ML algorithm



## 📂 Project Structure

📁 breast-cancer-prediction-model/
├── app.py
├── model.pkl
├── requirements.txt
└── README.md


## 🧠 ML Model

- **Model Used:** `XGBoostClassifier`
- **Library:** [xgboost](https://xgboost.readthedocs.io/)
- **Dataset:** [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Accuracy:** ~97%



## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
````

Or install manually:

```bash
pip install streamlit xgboost scikit-learn pandas numpy
```


## ▶️ Run Locally

streamlit run app.py



## 🛠 Technologies Used

* Python
* Streamlit
* XGBoost
* scikit-learn
* pandas
* NumPy


## 📦 Deployment

This app can be deployed using:

* **Streamlit Community Cloud**
* **GitHub + Streamlit Sharing**
* **Docker (optional)**

---

## 🙌 Acknowledgements

* [Streamlit](https://streamlit.io/)
* [XGBoost](https://xgboost.ai/)
* [Scikit-learn](https://scikit-learn.org/)
* [UCI Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)


