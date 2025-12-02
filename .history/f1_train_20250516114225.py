# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# import joblib

# from f1_data import setup_fastf1, fetch_session_data
# from f1_features import engineer_features


# def prepare_dataset(year: int, gp: str):
#     setup_fastf1("cache")
#     laps = fetch_session_data(year, gp, "R")
#     feats = engineer_features(laps)
#     # Placeholder target: replace with real results pulled from session.results
#     feats["position"] = feats.index + 1
#     return feats


# def train_model(df: pd.DataFrame, model_path: str = "f1_race_model.pkl"):
#     X = df.drop(["Driver", "position"], axis=1)
#     y = df["position"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")
#     return model


# if __name__ == "__main__":
#     df = prepare_dataset(2024, "Monza")
#     train_model(df)
