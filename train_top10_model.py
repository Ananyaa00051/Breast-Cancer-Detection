import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess dataset
data = pd.read_csv("data.csv")
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

y = data['diagnosis']
x_data = data.drop(['diagnosis'], axis=1)

# Top 10 selected features
selected_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_worst'
]

x_selected = x_data[selected_features]

# Normalize
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_selected)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.15, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
print("✅ Model trained on Top 10 features")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "xgb_top10_model.pkl")
joblib.dump(scaler, "scaler_top10.pkl")
print("🧠 Model saved as 'xgb_top10_model.pkl'")
print("🧪 Scaler saved as 'scaler_top10.pkl'")
