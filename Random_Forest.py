import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
dt = pd.read_csv("diabetes_prediction_dataset.csv")
data = pd.DataFrame(dt)

# Data Cleaning: Remove "Other" gender, drop duplicates & handle missing values
data = data[data["gender"] != "Other"]
data = data.drop_duplicates()
data = data.dropna()

# Encode categorical features
encoder = LabelEncoder()
data["gender"] = encoder.fit_transform(data["gender"])  # Female=0, Male=1
data["smoking_history"] = encoder.fit_transform(data["smoking_history"])

# Split features and target
X = data.drop("diabetes", axis=1)
y = data["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: Random Forest without tuning
model_default = RandomForestClassifier(random_state=42)
model_default.fit(X_train, y_train)

# Model 2: Random Forest with tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(model_default, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
model_tuned = grid_search.best_estimator_

# Save models
with open("random_forest_default.pkl", "wb") as f:
    pickle.dump(model_default, f)

with open("random_forest_tuned.pkl", "wb") as f:
    pickle.dump(model_tuned, f)

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# Evaluate both models
evaluate_model(model_default, X_test, y_test, "Random Forest Default")
evaluate_model(model_tuned, X_test, y_test, "Random Forest Tuned")
