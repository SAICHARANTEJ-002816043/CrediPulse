# CrediPulse
A machine learning solution for real-time credit risk prediction in banking.

## Overview
CrediPulse uses the UCI Credit Card Dataset to predict loan defaults with an end-to-end ML pipeline.

## Features
- **EDA**: Analysis in `notebooks/EDA.ipynb`.
- **Preprocessing**: `src/preprocess.py`.
- **Models**: XGBoost (ROC-AUC ~0.76) in `src/model.py`.
- **Interpretability**: SHAP (`XGBoost_shap_summary.png`).
- **Deployment**: Flask API in `src/api.py`.

## Setup
1. Clone: `git clone https://github.com/SAICHARANTEJ-002816043/CrediPulse.git`
2. Activate venv: `source venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Run EDA: Open `notebooks/EDA.ipynb`.
5. Train: `python src/model.py`
6. Start API: `python src/api.py` (port 5002)

## Results
- Achieved 76% ROC-AUC on 30,000 records.
- Predicted 32.32% default probability via API.

## Author
Sai Charan Tej - [linkedin.com/in/saicharantejk] | [GitHub: SAICHARANTEJ-002816043]