# Banking Credit Risk Predictor
An industry-level ML project to predict loan defaults in the banking sector.

## Overview
This project uses the UCI Credit Card Dataset to predict customer loan defaults with an end-to-end ML pipeline, including preprocessing, modeling, interpretability, and deployment.

## Features
- **EDA**: Detailed analysis in `notebooks/EDA.ipynb`.
- **Preprocessing**: Scaling and stratified splitting in `src/preprocess.py`.
- **Models**: Logistic Regression, Random Forest, XGBoost (ROC-AUC ~0.76) in `src/model.py`.
- **Interpretability**: SHAP analysis for XGBoost (`XGBoost_shap_summary.png`).
- **Deployment**: Flask API for real-time predictions in `src/api.py`.

## Setup
1. Clone: `git clone https://github.com/SAICHARANTEJ-002816043/banking-credit-risk-predictor.git`
2. Activate venv: `source venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Run EDA: Open `notebooks/EDA.ipynb` in Jupyter.
5. Train models: `python src/model.py`
6. Start API: `python src/api.py` (runs on port 5001; generates `model.pkl` and `scaler.pkl`)

## Results
- Achieved ~0.76 ROC-AUC with XGBoost on 30,000 records.
- Deployed a scalable API for real-time risk assessment (e.g., predicted 32.32% default probability for a sample input).

## Author
Sai Charan Tej - [LinkedIn] | [GitHub: SAICHARANTEJ-002816043]