import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt
from src.preprocess import load_and_preprocess_data

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        results[name] = {"model": model, "roc_auc": roc_auc}
        print(f"{name} - ROC-AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred))
    return results

def explain_model(model, X_test, model_name, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
    plt.savefig(f"{model_name}_shap_summary.png")
    plt.close()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data("/Users/charan/banking-credit-risk-predictor/data/credit_data.csv")
    results = train_models(X_train, X_test, y_train, y_test)
    # Use absolute path or predefine feature names
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'credit_data.csv')
    feature_names = pd.read_csv(data_path).drop(columns=["ID", "default payment next month"]).columns
    explain_model(results["XGBoost"]["model"], X_test, "XGBoost", feature_names)