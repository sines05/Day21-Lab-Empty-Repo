import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

EVAL_THRESHOLD = 0.70

def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    # 1. Read data
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Data Drift / Distribution
    class_counts = y_train.value_counts(normalize=True).to_dict()
    distribution = {str(k): float(v) for k, v in class_counts.items()}
    for cls, ratio in distribution.items():
        if ratio < 0.10:
            print(f"CẢNH BÁO: Lớp {cls} chỉ chiếm {ratio*100:.2f}% (< 10%) dữ liệu huấn luyện!")
            
    with mlflow.start_run():
        model_type = params.get("model_type", "random_forest")
        model_params = params.get(model_type, {})
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        # Bonus 2: Multiple Algorithms
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**model_params, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**model_params, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")
        
        # Bonus 3: Automated Performance Report
        prec = precision_score(y_eval, preds, average=None, zero_division=0).tolist()
        rec = recall_score(y_eval, preds, average=None, zero_division=0).tolist()
        cm = confusion_matrix(y_eval, preds).tolist()
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(f"Model: {model_type}\n")
            f.write("Confusion Matrix:\n")
            for row in cm:
                f.write(f"{row}\n")
            f.write("\nPrecision per class:\n")
            for i, p in enumerate(prec):
                f.write(f"Class {i}: {p:.4f}\n")
            f.write("\nRecall per class:\n")
            for i, r in enumerate(rec):
                f.write(f"Class {i}: {r:.4f}\n")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"[{model_type}] Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Metrics for CI/CD
        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc, 
                "f1_score": f1,
                "label_distribution": distribution
            }, f, indent=2)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
