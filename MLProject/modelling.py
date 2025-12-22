import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Konfigurasi logging mlflow ke DagsHub
DAGSHUB_REPO_OWNER = "ayutksnaa"      
DAGSHUB_REPO_NAME = "msml-repo" 
MLFLOW_TRACKING_URI = "https://dagshub.com/ayutksnaa/msml-repo.mlflow" 

dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, 'water_potability_preprocessing/train_clean_automated.csv')
TEST_PATH = os.path.join(BASE_DIR, 'water_potability_preprocessing/test_clean_automated.csv')

print("Memuat data...")
try:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
except FileNotFoundError:
    print(f"Error: File dataset tidak ditemukan di {TRAIN_PATH}.")
    exit()

X_train = train_df.drop('Potability', axis=1)
y_train = train_df['Potability']
X_test = test_df.drop('Potability', axis=1)
y_test = test_df['Potability']

# Hyperparameter Tuning dengan RandomizedSearchCV
print("Memulai Hyperparameter Tuning...")
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [50, 100], 
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=5,  
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
best_params = random_search.best_params_
print(f"Tuning Selesai! Params: {best_params}")

# Evaluasi
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Mlflow Logging
mlflow.sklearn.autolog(disable=True)
mlflow.set_experiment("Water_Potability_Advanced_CI")

print("Mengirim log ke DagsHub...")
with mlflow.start_run(run_name="RandomForest_Tuned_Advanced") as run:
    
    # Mengambil ID dari run ini dan simpan ke file
    run_id = run.info.run_id
    print(f"Run ID Detected: {run_id}")
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    # ===============================

    mlflow.log_params(best_params)
    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
    
    # Artefak 1: Confusion Matrix
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Artefak 2: Feature Importance
    plt.figure(figsize=(8,6))
    importances = best_model.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)[::-1]
    sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    # Log Model
    signature = infer_signature(X_train, y_pred)
    input_example = X_train.head(1)
    mlflow.sklearn.log_model(best_model, "model", signature=signature, input_example=input_example)
    
    # Bersihkan file
    if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")
    if os.path.exists("feature_importance.png"): os.remove("feature_importance.png")

print("Selesai! Run ID tersimpan di run_id.txt")