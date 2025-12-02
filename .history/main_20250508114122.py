# Installation Instructions for M1 Mac (in this repo's root README or as comments here):
# 1. Install Homebrew if you haven’t already:
#    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 2. Install Python 3.10 (or later):
#    brew install python@3.10
# 3. Create and activate a virtual environment:
#    python3 -m venv venv
#    source venv/bin/activate
# 4. Upgrade pip and install dependencies:
#    pip install --upgrade pip
#    pip install fastf1 pandas scikit-learn xgboost matplotlib

# -------- file: f1_data.py --------
import fastf1
from fastf1 import Cache
import pandas as pd


def setup_fastf1(cache_dir: str = "cache"):
    """
    Enable caching to speed up repeated requests.
    """
    Cache.enable_cache(cache_dir)


def fetch_session_data(year: int, gp: str, session: str) -> pd.DataFrame:
    """
    Fetch lap and weather data for a given session.
    Args:
        year: e.g., 2024
        gp: Grand Prix name, e.g., 'Monza'
        session: 'Q' (Qualifying), 'R' (Race)
    Returns:
        DataFrame of laps with weather info.
    """
    fastf1.Cache.enable_cache("cache")
    sess = fastf1.get_session(year, gp, session)
    sess.load(telemetry=False, laps=True, weather=True)
    laps = sess.laps
    # merge weather info onto laps
    weather = sess.weather_data
    laps = laps.merge(weather, left_on="Time", right_on="Time", how="left")
    return laps


# -------- file: f1_features.py --------
import pandas as pd


def engineer_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Creates aggregate race-level features for ML.
    """
    # average lap time
    avg_lap = laps.groupby("Driver")["LapTime"].mean().rename("avg_lap_time")
    # lap time consistency (std dev)
    std_lap = laps.groupby("Driver")["LapTime"].std().rename("std_lap_time")
    # number of pit stops
    pit_stops = (
        laps[laps["PitOutTime"].notnull()].groupby("Driver").size().rename("pit_count")
    )
    # merge into one DataFrame
    features = pd.concat([avg_lap, std_lap, pit_stops], axis=1).fillna(0)
    return features.reset_index()


# -------- file: f1_train.py --------
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
