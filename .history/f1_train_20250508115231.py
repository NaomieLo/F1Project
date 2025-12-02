import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from f1_data import fetch_session_data
from f1_features import engineer_features


def prepare_dataset(year: int, gp: str):
    laps = fetch_session_data(year, gp, "R")
    feats = engineer_features(laps)
    # target: finishing position from Ergast or FastF1 standings
    # placeholder: randomly generate target (replace with real data)
    feats["position"] = feats.index + 1
    return feats


def train_model(df: pd.DataFrame):
    """
    Train and evaluate a regression model to predict finishing position.
    """
    X = df.drop(["Driver", "position"], axis=1)
    y = df["position"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    return model


if __name__ == "__main__":
    df = prepare_dataset(2024, "Monza")
    model = train_model(df)
    # Save model
    import joblib

    joblib.dump(model, "f1_race_model.pkl")
