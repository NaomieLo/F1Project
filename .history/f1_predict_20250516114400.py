import fastf1
from fastf1 import Cache
import joblib
import pandas as pd
from f1_data import fetch_session_data
from f1_features import engineer_features


def predict_race(
    year: int, gp: str, model_path: str = "f1_race_model.pkl"
) -> pd.DataFrame:
    """
    Use qualifying or practice features to predict race finishing positions.
    """
    Cache.enable_cache("cache")
    # Fetch qualifying data
    laps = fetch_session_data(year, gp, "Q")
    feats = engineer_features(laps)

    model = joblib.load(model_path)
    X = feats.drop(["Driver"], axis=1)
    preds = model.predict(X)
    feats["predicted_position"] = preds.round().astype(int)
    return feats[["Driver", "predicted_position"]]


def get_actual_results(year: int, gp: str) -> pd.DataFrame:
    """
    Fetch actual race results via FastF1.
    """
    sess = fastf1.get_session(year, gp, "R")
    sess.load()  # load results
    results = sess.results[["Driver", "Position"]]
    results.columns = ["Driver", "actual_position"]
    return results


# if __name__ == "__main__":
#     year = 2025
#     gp = "Miami"
#     preds = predict_race(year, gp)
#     actuals = get_actual_results(year, gp)
#     comparison = preds.merge(actuals, on="Driver")
#     comparison["error"] = (
#         comparison["predicted_position"] - comparison["actual_position"]
#     )
#     print(comparison.sort_values("predicted_position"))
