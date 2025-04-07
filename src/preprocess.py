import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop(columns=["ID", "default payment next month"])
    y = data["default payment next month"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("/Users/charan/banking-credit-risk-predictor/data/credit_data.csv")
    print("Data preprocessed successfully!")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("Training labels distribution:\n", pd.Series(y_train).value_counts(normalize=True))
    print("Test labels distribution:\n", pd.Series(y_test).value_counts(normalize=True))