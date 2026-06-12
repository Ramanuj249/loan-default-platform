import pandas as pd
# import numpy as np
import sys
sys.path.append('.')
from src.data.preprocessing import preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "CatBoost": CatBoostClassifier(),
        "LightGBM": LGBMClassifier()
    }
    return models

def train_and_evaluate(X, y, model_name, model):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        mlflow.log_metrics(metrics)
        mlflow.log_param("model_name", model_name)
        mlflow.sklearn.log_model(model, "model")

        return metrics

if __name__ == "__main__":
    df = pd.read_csv("/Users/shivamramanuj/Python_projects/loan-default-platform/data/application_train.csv")

    X, y, scalar = preprocess_data(df)

    models = get_models()

    result = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        metrics = train_and_evaluate(X,y, model_name, model)
        result[model_name] = metrics
        print(f"{model_name}- ROC AUC: {metrics['roc_auc']:.4f}")

    print("\n--- Model Comparison ---")
    for model_name, metrics in result.items():
        print(f"{model_name}: ROC AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
