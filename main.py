import joblib
scaler = joblib.load("src/models/models/scaler.pkl")
print("Features expected:", scaler.n_features_in_)
print(scaler.feature_names_in_)