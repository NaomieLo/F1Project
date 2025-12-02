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

# In f1_predict.py
# ... (imports and function definitions remain the same) ...

if __name__ == "__main__":
    year = 2025
    gp = "Miami"  # Or the specific event name/number for Miami 2025
    model_path = "f1_race_model.pkl"

    print(f"Predicting race results for {gp} {year} using Qualifying data...")
    predictions_df = predict_race(year, gp, model_path=model_path)

    if predictions_df.empty:
        print("Could not generate predictions. Halting comparison.")
        return  # Exit if no predictions

    print("\nPredicted Finishing Positions (based on Qualifying data):")
    print(predictions_df.sort_values("predicted_position"))

    print(f"\nFetching actual race results for {gp} {year}...")
    actuals_df = get_actual_results(year, gp)

    if actuals_df.empty:
        print("Could not fetch actual results. Comparison cannot be made.")
        return  # Exit if no actuals

    print("\nActual Race Results:")
    print(actuals_df.sort_values("actual_position"))

    # Merge predictions with actuals
    # Ensure 'Driver' column is the common key and has consistent values (e.g., 'VER', 'HAM')
    comparison_df = predictions_df.merge(
        actuals_df, on="Driver", how="inner"
    )  # 'inner' ensures only drivers in both sets are compared

    if comparison_df.empty:
        print(
            "\nNo common drivers found between predictions and actual results. Check driver identifiers."
        )
        return

    # Calculate error
    # Ensure positions are numeric before subtraction
    comparison_df["predicted_position"] = pd.to_numeric(
        comparison_df["predicted_position"]
    )
    comparison_df["actual_position"] = pd.to_numeric(comparison_df["actual_position"])

    comparison_df["error"] = (
        comparison_df["predicted_position"] - comparison_df["actual_position"]
    )

    print("\nComparison (Predicted vs. Actual):")
    print(comparison_df.sort_values("predicted_position"))

    mae = comparison_df["error"].abs().mean()
    print(f"\nMean Absolute Error for {gp} {year}: {mae:.2f} positions")
