# import fastf1
# from fastf1 import Cache
# import joblib
# import pandas as pd
# from f1_data import fetch_session_data
# from f1_features import engineer_features


# def predict_race(
#     year: int, gp: str, model_path: str = "f1_race_model.pkl"
# ) -> pd.DataFrame:
#     """
#     Use qualifying or practice features to predict race finishing positions.
#     """
#     Cache.enable_cache("cache")
#     # Fetch qualifying data
#     laps = fetch_session_data(year, gp, "Q")
#     feats = engineer_features(laps)

#     model = joblib.load(model_path)
#     X = feats.drop(["Driver"], axis=1)
#     preds = model.predict(X)
#     feats["predicted_position"] = preds.round().astype(int)
#     return feats[["Driver", "predicted_position"]]


# def get_actual_results(year: int, gp: str) -> pd.DataFrame:
#     """
#     Fetch actual race results via FastF1.
#     """
#     sess = fastf1.get_session(year, gp, "R")
#     sess.load()  # load results
#     results = sess.results[["Driver", "Position"]]
#     results.columns = ["Driver", "actual_position"]
#     return results


# # if __name__ == "__main__":
# #     year = 2025
# #     gp = "Miami"
# #     preds = predict_race(year, gp)
# #     actuals = get_actual_results(year, gp)
# #     comparison = preds.merge(actuals, on="Driver")
# #     comparison["error"] = (
# #         comparison["predicted_position"] - comparison["actual_position"]
# #     )
# #     print(comparison.sort_values("predicted_position"))

# # In f1_predict.py
# # ... (imports and function definitions remain the same) ...

# if __name__ == "__main__":
#     year = 2025
#     gp = "Miami"  # Or the specific event name/number for Miami 2025
#     model_path = "f1_race_model.pkl"

#     print(f"Predicting race results for {gp} {year} using Qualifying data...")
#     predictions_df = predict_race(year, gp, model_path=model_path)

#     if predictions_df.empty:
#         print("Could not generate predictions. Halting comparison.")
#         return  # Exit if no predictions

#     print("\nPredicted Finishing Positions (based on Qualifying data):")
#     print(predictions_df.sort_values("predicted_position"))

#     print(f"\nFetching actual race results for {gp} {year}...")
#     actuals_df = get_actual_results(year, gp)

#     if actuals_df.empty:
#         print("Could not fetch actual results. Comparison cannot be made.")
#         return  # Exit if no actuals

#     print("\nActual Race Results:")
#     print(actuals_df.sort_values("actual_position"))

#     # Merge predictions with actuals
#     # Ensure 'Driver' column is the common key and has consistent values (e.g., 'VER', 'HAM')
#     comparison_df = predictions_df.merge(
#         actuals_df, on="Driver", how="inner"
#     )  # 'inner' ensures only drivers in both sets are compared

#     if comparison_df.empty:
#         print(
#             "\nNo common drivers found between predictions and actual results. Check driver identifiers."
#         )
#         return

#     # Calculate error
#     # Ensure positions are numeric before subtraction
#     comparison_df["predicted_position"] = pd.to_numeric(
#         comparison_df["predicted_position"]
#     )
#     comparison_df["actual_position"] = pd.to_numeric(comparison_df["actual_position"])

#     comparison_df["error"] = (
#         comparison_df["predicted_position"] - comparison_df["actual_position"]
#     )

#     print("\nComparison (Predicted vs. Actual):")
#     print(comparison_df.sort_values("predicted_position"))

#     mae = comparison_df["error"].abs().mean()
#     print(f"\nMean Absolute Error for {gp} {year}: {mae:.2f} positions")


# f1_predict.py
import fastf1
from fastf1 import Cache
import joblib
import pandas as pd
from f1_data import fetch_session_data  # Assuming this is in f1_data.py
from f1_features import engineer_features  # Assuming this is in f1_features.py


# def predict_race(
#     year: int, gp: any, model_path: str = "f1_race_model.pkl"
# ) -> pd.DataFrame:
#     """
#     Use qualifying features to predict race finishing positions.
#     'gp' can be the event name (e.g., "Miami Grand Prix") or round number.
#     """
#     Cache.enable_cache("cache")  # Ensure cache is enabled
#     print(f"Fetching Qualifying data for {gp}, {year} to make predictions...")
#     # Fetch qualifying data
#     try:
#         # 'Q' for Qualifying session
#         laps = fetch_session_data(year, gp, "Q")
#         if laps.empty:
#             print(
#                 f"No Qualifying lap data found for {gp}, {year}. Cannot make predictions."
#             )
#             return pd.DataFrame()
#     except Exception as e:
#         print(f"Error fetching qualifying session data for {gp}, {year}: {e}")
#         print(
#             "Please ensure the event name/round number is correct and the session has concluded."
#         )
#         return pd.DataFrame()

#     feats = engineer_features(laps)
#     if feats.empty:
#         print(f"Feature engineering failed or produced no data for {gp}, {year}.")
#         return pd.DataFrame()

#     try:
#         model = joblib.load(model_path)
#     except FileNotFoundError:
#         print(f"Error: Model file not found at {model_path}")
#         return pd.DataFrame()
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return pd.DataFrame()

#     X = feats.drop(
#         ["Driver"], axis=1, errors="ignore"
#     )  # errors='ignore' if 'Driver' might not be there

#     # Ensure columns match what the model was trained on
#     # This is a simple check; a more robust way is to save/load column order from training
#     # For now, assuming engineer_features consistently produces the same columns.
#     # If you encounter issues, ensure X's columns here match X_train from f1_train.py

#     preds = model.predict(X)
#     feats["predicted_position"] = preds.round().astype(int)
#     return feats[["Driver", "predicted_position"]]


def predict_race(
    year: int, gp: any, model_path: str = "f1_race_model.pkl"
) -> pd.DataFrame:
    Cache.enable_cache("cache")
    print(f"Fetching Qualifying data for {gp}, {year} to make predictions...")
    try:
        laps = fetch_session_data(year, gp, "Q")
        if laps.empty:
            print(
                f"No Qualifying lap data found for {gp}, {year}. Cannot make predictions."
            )
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching qualifying session data for {gp}, {year}: {e}")
        return pd.DataFrame()

    feats = engineer_features(
        laps
    )  # Assume this returns a DataFrame with 'Driver' and other features

    try:
        q_session = fastf1.get_session(year, gp, "Q")
        q_session.load(laps=False, telemetry=False, weather=False, messages=False)
        q_results = q_session.results

        if q_results is not None and not q_results.empty:
            q_positions = q_results[["Abbreviation", "Position"]].copy()
            q_positions.rename(
                columns={"Abbreviation": "Driver", "Position": "QualifyingPosition"},
                inplace=True,
            )
            q_positions["QualifyingPosition"] = pd.to_numeric(
                q_positions["QualifyingPosition"], errors="coerce"
            )

            feats = feats_from_laps.merge(q_positions, on="Driver", how="left")
            feats["QualifyingPosition"].fillna(20, inplace=True)
        else:
            print(
                f"Warning: Could not load Q results for {year} {gp} (prediction). Using default QPos."
            )
            feats = feats_from_laps.copy()
            feats["QualifyingPosition"] = 20
    except Exception as e:
        print(
            f"Error loading Q results for {year} {gp} (prediction): {e}. Using default QPos."
        )
        feats = feats_from_laps.copy()
        feats["QualifyingPosition"] = 20
    if feats.empty:
        print(f"Feature engineering failed or produced no data for {gp}, {year}.")
        return pd.DataFrame()

    # --- This is where you'll add the 'QualifyingPosition' feature later if you choose that improvement ---
    # For now, we'll stick to the current features for this unique rank fix.

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading model: {e}")
        return pd.DataFrame()

    # Prepare feature set X for prediction (ensure 'Driver' is not included if it was dropped during training)
    if "Driver" in feats.columns:
        X = feats.drop(["Driver"], axis=1)
    else:
        X = (
            feats.copy()
        )  # If engineer_features already returns only feature columns without Driver ID

    # Ensure X's columns match the order and names the model was trained on.
    # If you added/removed features during training, X here must match.

    raw_preds = model.predict(X)  # Get the raw continuous prediction scores

    # Create a new DataFrame for predictions or add to feats
    # For simplicity, let's re-use feats if it contains 'Driver'
    # If feats doesn't have 'Driver' because X was feats.copy(), you'll need to merge 'Driver' back or handle differently.
    # Assuming engineer_features gives 'Driver' and features, and feats still has 'Driver'.

    feats["predicted_raw_score"] = raw_preds

    # Assign unique ranks. Lower score = better position (1st, 2nd, etc.)
    # .rank(method='first') ensures that ties in score get unique ranks based on their order.
    feats["predicted_position"] = (
        feats["predicted_raw_score"].rank(method="first").astype(int)
    )

    return feats[["Driver", "predicted_position"]].sort_values("predicted_position")


def get_actual_results(year: int, gp: any) -> pd.DataFrame:
    """
    Fetch actual race results via FastF1.
    'gp' can be the event name (e.g., "Miami Grand Prix") or round number.
    """
    Cache.enable_cache("cache")  # Ensure cache is enabled
    print(f"Fetching actual Race results for {gp}, {year}...")
    try:
        sess = fastf1.get_session(year, gp, "R")  # 'R' for Race session
        # Load minimal data needed for results
        sess.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"Error fetching race session for {gp}, {year}: {e}")
        return pd.DataFrame()

    if sess.results is None or sess.results.empty:
        print(f"No actual race results found for {gp}, {year}.")
        return pd.DataFrame()

    # 'Abbreviation' is typically the driver identifier in session.results
    # 'Position' is the finishing position
    results = sess.results[["Abbreviation", "Position"]].copy()
    results.rename(
        columns={"Abbreviation": "Driver", "Position": "actual_position"}, inplace=True
    )
    results["actual_position"] = pd.to_numeric(
        results["actual_position"], errors="coerce"
    )
    results.dropna(
        subset=["actual_position"], inplace=True
    )  # Remove DNFs that might be NaN
    if not results.empty:
        results["actual_position"] = results["actual_position"].astype(int)
    return results


if __name__ == "__main__":
    # --- Configuration ---
    predict_year = 2025
    # For 'predict_gp', use the common name, official name, or round number.
    # FastF1 is usually good with "Miami". If it fails, try "Miami Grand Prix".
    # You can also find the exact event name or round number from:
    # import fastf1
    # schedule_2025 = fastf1.get_event_schedule(2025)
    # print(schedule_2025[['RoundNumber', 'EventName', 'EventDate']])
    predict_gp = "Miami"
    model_file_path = "f1_race_model.pkl"

    print(f"--- F1 Prediction for {predict_gp} {predict_year} ---")

    # --- Stage 1: Make Predictions (Run this after Qualifying) ---
    print("\nSTAGE 1: Generating Predictions (based on Qualifying data)")
    predicted_positions = predict_race(
        predict_year, predict_gp, model_path=model_file_path
    )

    if not predicted_positions.empty:
        print("\nPredicted Finishing Positions:")
        print(predicted_positions.sort_values("predicted_position"))
    else:
        print("\nCould not generate predictions. Please check logs.")

    # --- Stage 2: Compare with Actual Results (Run this AFTER the Race) ---
    # You can comment out this section if you are running this script before the race has finished.
    # Or, run the whole script after the race.

    print("\n\nSTAGE 2: Comparing with Actual Race Results")
    print(
        "(This part will only work meaningfully AFTER the race has concluded and results are available)"
    )

    actual_race_results = get_actual_results(predict_year, predict_gp)

    if not predicted_positions.empty and not actual_race_results.empty:
        print("\nActual Race Results:")
        print(actual_race_results.sort_values("actual_position"))

        comparison_df = predicted_positions.merge(
            actual_race_results, on="Driver", how="inner"
        )

        if not comparison_df.empty:
            # Ensure positions are numeric before subtraction
            comparison_df["predicted_position"] = pd.to_numeric(
                comparison_df["predicted_position"]
            )
            comparison_df["actual_position"] = pd.to_numeric(
                comparison_df["actual_position"]
            )

            comparison_df["error"] = (
                comparison_df["predicted_position"] - comparison_df["actual_position"]
            )
            print("\nComparison (Predicted vs. Actual):")
            print(comparison_df.sort_values("predicted_position"))

            mae = comparison_df["error"].abs().mean()
            print(
                f"\nMean Absolute Error for {predict_gp} {predict_year}: {mae:.2f} positions"
            )
        else:
            print(
                "\nCould not merge predictions with actual results. Check driver identifiers or if any drivers from qualifying participated in the race."
            )
    elif predicted_positions.empty:
        print("\nCannot compare because predictions were not generated.")
    else:
        print(
            "\nActual race results not found or empty. Cannot perform comparison yet."
        )
