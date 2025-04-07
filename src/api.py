import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import load_and_preprocess_data
from src.model import train_models
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Train and save model
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'credit_data.csv')
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
results = train_models(X_train, X_test, y_train, y_test)
model = results["XGBoost"]["model"]
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [data["features"]]
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]
    return jsonify({"default_probability": float(prob)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)  # Changed to 5001