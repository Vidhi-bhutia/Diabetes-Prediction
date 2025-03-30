import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# Load models
def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

models = {
    "LightGBM (Default)": load_model("lightgbm_default.pkl"),
    "LightGBM (Tuned)": load_model("lightgbm_tuned.pkl"),
    "XGBoost (Default)": load_model("xgboost_default.pkl"),
    "XGBoost (Tuned)": load_model("xgboost_tuned.pkl"),
    "Random Forest (Default)": load_model("random_forest_default.pkl"),
    "Random Forest (Tuned)": load_model("random_forest_tuned.pkl"),
}

# Load dataset
dt = pd.read_csv("diabetes_prediction_dataset.csv")
data = dt.copy()

# Data Cleaning
data = data[data["gender"] != "Other"].drop_duplicates().dropna()
data["gender"] = data["gender"].map({"Female": 0, "Male": 1})
data["smoking_history"] = data["smoking_history"].map({"never": 0, "No Info": 1, "former": 2, "current": 3})

X = data.drop("diabetes", axis=1)
y = data["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stTextInput, .stNumberInput, .stSelectbox, .stButton {
        background-color: #333;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 Diabetes Prediction App")
st.write("Fill in the details to predict diabetes status using multiple ML models.")

# Input Form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox("Smoking History", ["never", "No Info", "former", "current"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

if st.button("Predict Diabetes"):
    input_data = np.array([[
        1 if gender == "Male" else 0, age, hypertension, heart_disease,
        {"never": 0, "No Info": 1, "former": 2, "current": 3}[smoking_history],
        bmi, HbA1c_level, blood_glucose_level
    ]])
    
    results = {}
    for name, model in models.items():
        prediction = model.predict(input_data)[0]
        results[name] = "Diabetic" if prediction == 1 else "Non-Diabetic"
    
    st.subheader("🩺 Prediction Results:")
    for model, result in results.items():
        st.write(f"✔️ **{model}:** {result}")

    # Model Accuracy Visualization
    accuracies = {name: accuracy_score(y_test, model.predict(X_test)) for name, model in models.items()}
    
    st.subheader("📊 Model Performance:")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), ax=ax, palette="coolwarm")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader("🔍 ROC Curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)
