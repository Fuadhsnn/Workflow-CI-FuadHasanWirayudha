# ===============================
# Modelling & Hyperparameter Tuning (ANTI RUN NOT FOUND)
# ===============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import mlflow
import dagshub

# ===============================
# HARD RESET MLFLOW ENV (INI KUNCI)
# ===============================
os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

mlflow.end_run()

# ===============================
# Setup DagsHub + MLflow
# ===============================
dagshub.init(
    repo_owner="Fuadhsnn",
    repo_name="Eksperimen_SML_FuadHasanWirayudha",
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/Fuadhsnn/Eksperimen_SML_FuadHasanWirayudha.mlflow")
mlflow.set_experiment("Diabetes_Prediction_Fuad")

# ===============================
# Load Dataset
# ===============================
DATA_PATH = "cleaned_pima_diabetes_fuad.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("Outcome", axis=1)
y = df["Outcome"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Optuna Hyperparameter Tuning
# ===============================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
best_params.update({"random_state": 42, "n_jobs": -1})

# ===============================
# Train Final Model
# ===============================
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ===============================
# Metrics
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ===============================
# MLflow Logging (SAFE MODE)
# ===============================
with mlflow.start_run(
    run_name="RandomForest_Optuna_Diabetes",
    nested=False
):

    for k, v in best_params.items():
        mlflow.log_param(k, v)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # Feature Importance
    fi_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=fi_df)
    plt.title("Feature Importance")
    plt.tight_layout()

    plt.savefig("feature_importance.png")
    plt.close()
    mlflow.log_artifact("feature_importance.png")

    # Save Model
    joblib.dump(model, "diabetes_model.pkl")
    mlflow.log_artifact("diabetes_model.pkl")

print("✅ SUCCESS: MLflow logging ke DagsHub BERHASIL TANPA ERROR")
