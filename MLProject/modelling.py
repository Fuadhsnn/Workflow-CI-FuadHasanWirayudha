# ===============================
# CI Modelling + Optuna + MLflow (DagsHub Tracking, NON-INTERACTIVE)
# ===============================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import mlflow


# ===============================
# 1) MLFLOW AUTH (FROM ENV)
# ===============================
# GitHub Actions will provide these env vars via secrets
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")

# REQUIRED by MLflow REST auth
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Reset any leftover run id (prevents "Run not found")
os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
try:
    mlflow.end_run()
except:
    pass

# Tracking URI (repo mlflow)
mlflow.set_tracking_uri("https://dagshub.com/Fuadhsnn/Eksperimen_SML_FuadHasanWirayudha.mlflow")
mlflow.set_experiment("Diabetes_Prediction_Fuad_CI")


# ===============================
# 2) LOAD DATASET
# ===============================
DATA_PATH = "cleaned_pima_diabetes_fuad.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("Outcome", axis=1)
y = df["Outcome"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ===============================
# 3) OPTUNA TUNING
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
# 4) TRAIN FINAL MODEL
# ===============================
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# ===============================
# 5) LOGGING TO MLFLOW
# ===============================
with mlflow.start_run(run_name="RF_Optuna_Diabetes_CI"):

    # params
    for k, v in best_params.items():
        mlflow.log_param(k, v)

    # metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Confusion matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Feature importance artifact
    fi_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=fi_df)
    plt.title("Feature Importance")
    plt.tight_layout()
    fi_path = "feature_importance.png"
    plt.savefig(fi_path)
    plt.close()
    mlflow.log_artifact(fi_path)

    # Save model artifact
    model_path = "diabetes_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

print("✅ SUCCESS: CI training + logging to DagsHub completed")
