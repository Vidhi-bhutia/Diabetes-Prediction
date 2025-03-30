import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load models
with open("lightgbm_default.pkl", "rb") as f:
    lgb_model_default = pickle.load(f)

with open("lightgbm_tuned.pkl", "rb") as f:
    lgb_model_tuned = pickle.load(f)

with open("xgboost_default.pkl", "rb") as f:
    xgb_model_default = pickle.load(f)

with open("xgboost_tuned.pkl", "rb") as f:
    xgb_model_tuned = pickle.load(f)
    
with open("random_forest_default.pkl", "rb") as f:
    rf_model_default = pickle.load(f)
    
with open("random_forest_tuned.pkl", "rb") as f:
    rf_model_tuned = pickle.load(f)

# Load dataset
dt = pd.read_csv("diabetes_prediction_dataset.csv")
data = pd.DataFrame(dt)

# Data Cleaning: Remove "Other" gender, drop duplicates & handle missing values
data = data[data["gender"] != "Other"]
data = data.drop_duplicates()
data = data.dropna()

# Encode categorical features
data["gender"] = data["gender"].map({"Female": 0, "Male": 1})
smoking_dict = {"never": 0, "No Info": 1, "former": 2, "current": 3}
data["smoking_history"] = data["smoking_history"].map(smoking_dict)

# Split dataset into training and testing sets
X = data.drop("diabetes", axis=1)
y = data["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Function to take user input
def get_user_input():
    print("\nEnter patient details for diabetes prediction:")
    gender = input("Gender (Male/Female): ").strip().lower()
    age = int(input("Age: "))
    hypertension = int(input("Hypertension (1-Yes, 0-No): "))
    heart_disease = int(input("Heart Disease (1-Yes, 0-No): "))
    smoking_history = input("Smoking History (never, No Info, former, current): ").strip().lower()
    bmi = float(input("BMI: "))
    HbA1c_level = float(input("HbA1c Level: "))
    blood_glucose_level = int(input("Blood Glucose Level: "))

    # Encoding inputs
    gender = 1 if gender == "male" else 0
    smoking_history = smoking_dict.get(smoking_history, 0)  # Default to 0 (never) if unknown

    return np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])

# Get input and predict
user_data = get_user_input()
pred_lgb_default = lgb_model_default.predict(user_data)[0]
pred_lgb_tuned = lgb_model_tuned.predict(user_data)[0]
pred_xgb_default = xgb_model_default.predict(user_data)[0]
pred_xgb_tuned = xgb_model_tuned.predict(user_data)[0]
pred_rf_default = rf_model_default.predict(user_data)[0]
pred_rf_tuned = rf_model_tuned.predict(user_data)[0]

# Evaluate models
y_pred_lgb_default = lgb_model_default.predict(X_test)
y_pred_lgb_tuned = lgb_model_tuned.predict(X_test)
y_pred_xgb_default = xgb_model_default.predict(X_test)
y_pred_xgb_tuned = xgb_model_tuned.predict(X_test)
y_pred_rf_default = rf_model_default.predict(X_test)
y_pred_rf_tuned = rf_model_tuned.predict(X_test)

accuracy_lgb_default = accuracy_score(y_test, y_pred_lgb_default)
accuracy_lgb_tuned = accuracy_score(y_test, y_pred_lgb_tuned)
accuracy_xgb_default = accuracy_score(y_test, y_pred_xgb_default)
accuracy_xgb_tuned = accuracy_score(y_test, y_pred_xgb_tuned)
accuracy_rf_default = accuracy_score(y_test, y_pred_rf_default)
accuracy_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)

# Print predictions and accuracy
print("\nDiabetes Prediction:")
print(f"🔹 LightGBM (Default): {'Diabetic' if pred_lgb_default == 1 else 'Non-Diabetic'} | Accuracy: {accuracy_lgb_default:.4f}")
print(f"🔹 LightGBM (Tuned): {'Diabetic' if pred_lgb_tuned == 1 else 'Non-Diabetic'} | Accuracy: {accuracy_lgb_tuned:.4f}")
print(f"🔹 XGBoost (Default): {'Diabetic' if pred_xgb_default == 1 else 'Non-Diabetic'} | Accuracy: {accuracy_xgb_default:.4f}")
print(f"🔹 XGBoost (Tuned): {'Diabetic' if pred_xgb_tuned == 1 else 'Non-Diabetic'} | Accuracy: {accuracy_xgb_tuned:.4f}")
print(f"🔹 Random Forest (Default): {'Diabetic' if pred_rf_default == 1 else 'Non-Diabetic'} | Accuracy: {accuracy_rf_default:.4f}")
print(f"🔹 Random Forest (Tuned): {'Diabetic' if pred_rf_tuned == 1 else 'Non-Diabetic'} | Accuracy: {accuracy_rf_tuned:.4f}")