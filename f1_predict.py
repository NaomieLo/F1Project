# import fastf1
# from fastf1 import Cache # Cache is now directly available under fastf1
# import joblib
# import pandas as pd
# import numpy as np # For np.number if needed in preprocessing

# # Assuming these are in your project structure and work correctly
# from f1_data import setup_fastf1, fetch_session_data
# from f1_features import engineer_features

# # Disable pandas SettingWithCopyWarning for cleaner output
# pd.options.mode.chained_assignment = None

# def preprocess_for_prediction(X_predict: pd.DataFrame, model_features: list) -> pd.DataFrame:
#     """
#     Ensures X_predict has the same columns (and order) as the training data X_train.
#     Handles one-hot encoding for categorical features and imputes NaNs.
#     Adds missing columns with default values (e.g., 0 for numeric, appropriate for OHE).
#     """
#     # --- Handle categorical features with one-hot encoding ---
#     categorical_cols = X_predict.select_dtypes(include=['object', 'category']).columns
#     if not categorical_cols.empty:
#         print(f"Preprocessing: Applying one-hot encoding to: {list(categorical_cols)}")
#         X_predict = pd.get_dummies(X_predict, columns=categorical_cols, dummy_na=False)

#     # --- Align columns with model_features ---
#     # Add missing columns (that were in training but not in current predict set) with 0
#     missing_cols = set(model_features) - set(X_predict.columns)
#     for c in missing_cols:
#         print(f"Preprocessing: Adding missing column '{c}' with default value 0.")
#         X_predict[c] = 0 # Or appropriate default (e.g., False for boolean OHE columns)

#     # Remove extra columns (in current predict set but not in training)
#     extra_cols = set(X_predict.columns) - set(model_features)
#     if extra_cols:
#         print(f"Preprocessing: Removing extra columns: {list(extra_cols)}")
#         X_predict = X_predict.drop(columns=list(extra_cols))

#     # Ensure order of columns is the same as when the model was trained
#     X_predict = X_predict[model_features]

#     # --- Imputation for remaining NaNs in numeric columns (if any) ---
#     numeric_cols_with_na = X_predict.select_dtypes(include=np.number).isnull().any()
#     cols_to_impute = numeric_cols_with_na[numeric_cols_with_na].index
#     if not cols_to_impute.empty:
#         print(f"Preprocessing: Imputing NaNs in numeric columns: {list(cols_to_impute)} with median...")
#         for col in cols_to_impute:
#             # IMPORTANT: Use median from TRAINING data if available, or calculate from current data
#             # For simplicity here, using median from current prediction data.
#             # Better: save training medians and use them.
#             median_val = X_predict[col].median()
#             X_predict[col].fillna(median_val, inplace=True)
#             print(f"Imputed NaNs in '{col}' with median: {median_val}")

#     return X_predict


# def predict_race(
#     year: int, gp_identifier: any, model_path: str = "f1_race_model.pkl"
# ) -> pd.DataFrame:
#     """
#     Uses Qualifying features to predict race finishing positions.
#     'gp_identifier' can be the event name or round number.
#     """
#     # setup_fastf1("cache") # Or fastf1.Cache.enable_cache("cache")
#     fastf1.Cache.enable_cache("cache") # More direct

#     print(f"Fetching Qualifying lap data for {gp_identifier}, {year} to make predictions...")
#     try:
#         # fetch_session_data MUST call session.load() internally before returning laps
#         laps = fetch_session_data(year, gp_identifier, "Q")
#         if laps.empty:
#             print(f"No Qualifying lap data found for {gp_identifier}, {year}. Cannot make predictions.")
#             return pd.DataFrame()
#     except Exception as e:
#         print(f"Error fetching qualifying lap data for {gp_identifier}, {year}: {e}")
#         return pd.DataFrame()

#     # Engineer base features from qualifying laps
#     # This should return a DataFrame with 'Driver' and other engineered features
#     feats_pred = engineer_features(laps)
#     if feats_pred.empty:
#         print(f"Feature engineering (from laps) failed or produced no data for {gp_identifier}, {year}.")
#         return pd.DataFrame()

#     # Add 'QualifyingPosition' feature, similar to how it's done in f1_train.py
#     print(f"Fetching Qualifying session results for 'QualifyingPosition': {year}, GP ID: {gp_identifier}...")
#     try:
#         q_session = fastf1.get_session(year, gp_identifier, "Q")
#         q_session.load(laps=False, telemetry=False, weather=False, messages=False)
#         q_results = q_session.results

#         if q_results is not None and not q_results.empty and 'Abbreviation' in q_results.columns and 'Position' in q_results.columns:
#             q_positions = q_results[["Abbreviation", "Position"]].copy()
#             q_positions.rename(
#                 columns={"Abbreviation": "Driver", "Position": "QualifyingPosition"},
#                 inplace=True,
#             )
#             q_positions["QualifyingPosition"] = pd.to_numeric(
#                 q_positions["QualifyingPosition"], errors="coerce"
#             )
#             feats_pred = feats_pred.merge(q_positions, on="Driver", how="left")
#         else:
#             print(f"Warning: Could not load Q results or 'Abbreviation'/'Position' missing for {year} {gp_identifier} (prediction). Defaulting QualifyingPosition.")
#             feats_pred = feats_pred.copy() # Ensure it's a copy before adding column
#             feats_pred["QualifyingPosition"] = 25 # Default high value
#     except Exception as e:
#         print(f"Error loading Q results for {year} {gp_identifier} (prediction): {e}. Defaulting QualifyingPosition.")
#         feats_pred = feats_pred.copy()
#         feats_pred["QualifyingPosition"] = 25

#     feats_pred["QualifyingPosition"].fillna(25, inplace=True) # Fill any remaining NaNs

#     if 'Driver' not in feats_pred.columns:
#         print("CRITICAL: 'Driver' column is missing from feats_pred. Cannot proceed.")
#         return pd.DataFrame()

#     # --- Load Model and Prepare Features ---
#     try:
#         model_bundle = joblib.load(model_path) # Expecting a dict: {'model': model_object, 'features': list_of_feature_names}
#         model = model_bundle['model']
#         model_training_features = model_bundle['features']
#     except FileNotFoundError:
#         print(f"Error: Model file not found at {model_path}")
#         return pd.DataFrame()
#     except KeyError:
#         print(f"Error: Model file {model_path} does not contain 'model' and 'features' keys. Please retrain and save with f1_train.py.")
#         # Fallback for older models that didn't save feature list (less robust)
#         print("Attempting to load model directly, but feature alignment might be an issue.")
#         try:
#             model = joblib.load(model_path)
#             model_training_features = None # Cannot verify features
#             print("WARNING: Loaded model without feature list. Predictions might be unreliable if feature sets differ.")
#         except Exception as e_load:
#             print(f"Error loading model directly: {e_load}")
#             return pd.DataFrame()
#     except Exception as e:
#         print(f"Error loading model bundle: {e}")
#         return pd.DataFrame()

#     # Prepare feature set X for prediction
#     drivers_for_prediction = feats_pred["Driver"].copy()
#     X_predict = feats_pred.drop(["Driver"], axis=1)

#     if model_training_features:
#         X_predict = preprocess_for_prediction(X_predict, model_training_features)
#     else:
#         print("WARNING: Proceeding without explicit feature alignment as model_training_features list was not available.")
#         # Basic check: if model has feature_names_in_ (sklearn >= 0.24)
#         if hasattr(model, 'feature_names_in_'):
#             try:
#                 X_predict = X_predict[model.feature_names_in_]
#             except KeyError as e_key:
#                 print(f"Feature mismatch: Model expects features {model.feature_names_in_}, but got {X_predict.columns}. Error: {e_key}")
#                 print("Please ensure engineer_features() and preprocessing create the correct features.")
#                 return pd.DataFrame()
#         else: # Older sklearn or different model type
#              print("Model does not have feature_names_in_. Cannot guarantee feature order. Predictions may fail or be inaccurate.")


#     # --- Make Predictions ---
#     try:
#         raw_preds = model.predict(X_predict)
#     except ValueError as ve:
#         print(f"ValueError during prediction: {ve}")
#         print("This often means the features in X_predict do not match what the model was trained on.")
#         print(f"Features expected by model (if available): {getattr(model, 'feature_names_in_', 'N/A')}")
#         print(f"Features provided for prediction: {X_predict.columns.tolist()}")
#         return pd.DataFrame()


#     # Combine drivers with their raw scores
#     predictions_df = pd.DataFrame({'Driver': drivers_for_prediction, 'predicted_raw_score': raw_preds})

#     # Assign unique ranks. Lower score = better predicted position.
#     predictions_df["predicted_position"] = (
#         predictions_df["predicted_raw_score"].rank(method="first").astype(int)
#     )

#     return predictions_df[["Driver", "predicted_position"]].sort_values("predicted_position")


# def get_actual_results(year: int, gp_identifier: any) -> pd.DataFrame:
#     """
#     Fetch actual race results via FastF1.
#     'gp_identifier' can be the event name or round number.
#     """
#     # setup_fastf1("cache") # Or fastf1.Cache.enable_cache("cache")
#     fastf1.Cache.enable_cache("cache")

#     print(f"Fetching actual Race results for {gp_identifier}, {year}...")
#     try:
#         sess = fastf1.get_session(year, gp_identifier, "R")
#         sess.load(laps=False, telemetry=False, weather=False, messages=False) # Load only what's needed for results
#     except Exception as e:
#         print(f"Error fetching race session for actual results {gp_identifier}, {year}: {e}")
#         return pd.DataFrame()

#     if sess.results is None or sess.results.empty:
#         print(f"No actual race results found for {gp_identifier}, {year}.")
#         return pd.DataFrame()

#     if 'Abbreviation' not in sess.results.columns or 'Position' not in sess.results.columns:
#         print(f"Warning: 'Abbreviation' or 'Position' missing in actual race results for {gp_identifier}, {year}.")
#         return pd.DataFrame()

#     results = sess.results[["Abbreviation", "Position"]].copy()
#     results.rename(
#         columns={"Abbreviation": "Driver", "Position": "actual_position"}, inplace=True
#     )
#     results["actual_position"] = pd.to_numeric(
#         results["actual_position"], errors="coerce"
#     )
#     results.dropna(subset=["actual_position"], inplace=True) # Remove DNFs etc.
#     if not results.empty:
#         results["actual_position"] = results["actual_position"].astype(int)
#     return results


# if __name__ == "__main__":
#     # --- Configuration ---
#     # Determine the upcoming weekend's race.
#     # For this example, let's manually set it.
#     # In a real scenario, you might fetch the schedule and find the next event.
#     predict_year = 2025 # Assuming today is before Emilia Romagna 2025
#     predict_gp_name = "Emilia Romagna Grand Prix" # Use the official event name from fastf1.get_event_schedule()
#     # Or use round number if known and more reliable:
#     # predict_gp_identifier = 7 # Example: If Emilia Romagna is Round 7 in 2025

#     # To find the correct identifier for "Emilia Romagna":
#     # import fastf1
#     # schedule_2025 = fastf1.get_event_schedule(2025)
#     # print(schedule_2025[schedule_2025['EventName'].str.contains("Emilia Romagna", case=False)])
#     # From that, you can get the RoundNumber or confirm EventName.
#     # Let's assume we found Round 7 is "Emilia Romagna Grand Prix"
#     predict_gp_identifier = predict_gp_name # Or the round number e.g. 7

#     model_file_path = "f1_race_model.pkl"

#     print(f"--- F1 Prediction for {predict_gp_identifier} {predict_year} ---")

#     # --- Stage 1: Make Predictions (Run this after Qualifying) ---
#     print("\nSTAGE 1: Generating Predictions (based on Qualifying data)")
#     predicted_positions = predict_race(
#         predict_year, predict_gp_identifier, model_path=model_file_path
#     )

#     if not predicted_positions.empty:
#         print("\nPredicted Finishing Positions:")
#         print(predicted_positions) # Already sorted by predict_race
#     else:
#         print("\nCould not generate predictions. Please check logs.")

#     # --- Stage 2: Compare with Actual Results (Run this AFTER the Race) ---
#     print("\n\nSTAGE 2: Comparing with Actual Race Results")
#     print("(This part will only work meaningfully AFTER the race has concluded and results are available)")

#     actual_race_results = get_actual_results(predict_year, predict_gp_identifier)

#     if not predicted_positions.empty and not actual_race_results.empty:
#         print("\nActual Race Results:")
#         print(actual_race_results.sort_values("actual_position"))

#         comparison_df = predicted_positions.merge(
#             actual_race_results, on="Driver", how="inner" # Inner join on common drivers
#         )

#         if not comparison_df.empty:
#             comparison_df["error"] = (
#                 comparison_df["predicted_position"] - comparison_df["actual_position"]
#             )
#             print("\nComparison (Predicted vs. Actual):")
#             # Sort by predicted position for consistency, or actual_position if preferred
#             print(comparison_df.sort_values("predicted_position"))

#             mae = comparison_df["error"].abs().mean()
#             print(
#                 f"\nMean Absolute Error for {predict_gp_identifier} {predict_year}: {mae:.2f} positions"
#             )
#         else:
#             print("\nCould not merge predictions with actual results. (No common drivers or one set empty).")
#     elif predicted_positions.empty:
#         print("\nCannot compare because predictions were not generated.")
#     else: # actual_race_results is empty but predictions were generated
#         print("\nActual race results not found or empty. Cannot perform comparison yet.")

import fastf1
from fastf1 import Cache  # Cache is now directly available under fastf1
import joblib
import pandas as pd
import numpy as np  # For np.number if needed in preprocessing

# Assuming these are in your project structure and work correctly
from f1_data import setup_fastf1, fetch_session_data
from f1_features import engineer_features

# Disable pandas SettingWithCopyWarning for cleaner output
pd.options.mode.chained_assignment = None


def preprocess_for_prediction(
    X_predict: pd.DataFrame, model_features: list
) -> pd.DataFrame:
    """
    Ensures X_predict has the same columns (and order) as the training data X_train.
    Handles one-hot encoding for categorical features and imputes NaNs.
    Adds missing columns with default values (e.g., 0 for numeric, appropriate for OHE).
    """
    # --- Handle categorical features with one-hot encoding ---
    categorical_cols = X_predict.select_dtypes(include=["object", "category"]).columns
    if not categorical_cols.empty:
        print(f"Preprocessing: Applying one-hot encoding to: {list(categorical_cols)}")
        X_predict = pd.get_dummies(X_predict, columns=categorical_cols, dummy_na=False)

    # --- Align columns with model_features ---
    # Add missing columns (that were in training but not in current predict set) with 0
    missing_cols = set(model_features) - set(X_predict.columns)
    for c in missing_cols:
        print(f"Preprocessing: Adding missing column '{c}' with default value 0.")
        X_predict[c] = 0  # Or appropriate default (e.g., False for boolean OHE columns)

    # Remove extra columns (in current predict set but not in training)
    extra_cols = set(X_predict.columns) - set(model_features)
    if extra_cols:
        print(f"Preprocessing: Removing extra columns: {list(extra_cols)}")
        X_predict = X_predict.drop(columns=list(extra_cols))

    # Ensure order of columns is the same as when the model was trained
    X_predict = X_predict[model_features]

    # --- Imputation for remaining NaNs in numeric columns (if any) ---
    numeric_cols_with_na = X_predict.select_dtypes(include=np.number).isnull().any()
    cols_to_impute = numeric_cols_with_na[numeric_cols_with_na].index
    if not cols_to_impute.empty:
        print(
            f"Preprocessing: Imputing NaNs in numeric columns: {list(cols_to_impute)} with median..."
        )
        for col in cols_to_impute:
            # IMPORTANT: Use median from TRAINING data if available, or calculate from current data
            # For simplicity here, using median from current prediction data.
            # Better: save training medians and use them.
            median_val = X_predict[col].median()
            X_predict[col].fillna(median_val, inplace=True)
            print(f"Imputed NaNs in '{col}' with median: {median_val}")

    return X_predict


def predict_race(
    year: int, gp_identifier: any, model_path: str = "f1_race_model.pkl"
) -> pd.DataFrame:
    """
    Uses Qualifying features to predict race finishing positions.
    'gp_identifier' can be the event name or round number.
    """
    # setup_fastf1("cache") # Or fastf1.Cache.enable_cache("cache")
    fastf1.Cache.enable_cache("cache")  # More direct

    print(
        f"Fetching Qualifying lap data for {gp_identifier}, {year} to make predictions..."
    )
    try:
        # fetch_session_data MUST call session.load() internally before returning laps
        laps = fetch_session_data(year, gp_identifier, "Q")
        if laps.empty:
            print(
                f"No Qualifying lap data found for {gp_identifier}, {year}. Cannot make predictions."
            )
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching qualifying lap data for {gp_identifier}, {year}: {e}")
        return pd.DataFrame()

    # Engineer base features from qualifying laps
    # This should return a DataFrame with 'Driver' and other engineered features
    feats_pred = engineer_features(laps)
    if feats_pred.empty:
        print(
            f"Feature engineering (from laps) failed or produced no data for {gp_identifier}, {year}."
        )
        return pd.DataFrame()

    # Add 'QualifyingPosition' feature, similar to how it's done in f1_train.py
    print(
        f"Fetching Qualifying session results for 'QualifyingPosition': {year}, GP ID: {gp_identifier}..."
    )
    try:
        q_session = fastf1.get_session(year, gp_identifier, "Q")
        q_session.load(laps=False, telemetry=False, weather=False, messages=False)
        q_results = q_session.results

        if (
            q_results is not None
            and not q_results.empty
            and "Abbreviation" in q_results.columns
            and "Position" in q_results.columns
        ):
            q_positions = q_results[["Abbreviation", "Position"]].copy()
            q_positions.rename(
                columns={"Abbreviation": "Driver", "Position": "QualifyingPosition"},
                inplace=True,
            )
            q_positions["QualifyingPosition"] = pd.to_numeric(
                q_positions["QualifyingPosition"], errors="coerce"
            )
            feats_pred = feats_pred.merge(q_positions, on="Driver", how="left")
        else:
            print(
                f"Warning: Could not load Q results or 'Abbreviation'/'Position' missing for {year} {gp_identifier} (prediction). Defaulting QualifyingPosition."
            )
            feats_pred = feats_pred.copy()  # Ensure it's a copy before adding column
            feats_pred["QualifyingPosition"] = 25  # Default high value
    except Exception as e:
        print(
            f"Error loading Q results for {year} {gp_identifier} (prediction): {e}. Defaulting QualifyingPosition."
        )
        feats_pred = feats_pred.copy()
        feats_pred["QualifyingPosition"] = 25

    feats_pred["QualifyingPosition"].fillna(25, inplace=True)  # Fill any remaining NaNs

    if "Driver" not in feats_pred.columns:
        print("CRITICAL: 'Driver' column is missing from feats_pred. Cannot proceed.")
        return pd.DataFrame()

    # --- Load Model and Prepare Features ---
    try:
        model_bundle = joblib.load(
            model_path
        )  # Expecting a dict: {'model': model_object, 'features': list_of_feature_names}
        model = model_bundle["model"]
        model_training_features = model_bundle["features"]
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return pd.DataFrame()
    except KeyError:
        print(
            f"Error: Model file {model_path} does not contain 'model' and 'features' keys. Please retrain and save with f1_train.py."
        )
        # Fallback for older models that didn't save feature list (less robust)
        print(
            "Attempting to load model directly, but feature alignment might be an issue."
        )
        try:
            model = joblib.load(model_path)
            model_training_features = None  # Cannot verify features
            print(
                "WARNING: Loaded model without feature list. Predictions might be unreliable if feature sets differ."
            )
        except Exception as e_load:
            print(f"Error loading model directly: {e_load}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading model bundle: {e}")
        return pd.DataFrame()

    # Prepare feature set X for prediction
    drivers_for_prediction = feats_pred["Driver"].copy()
    X_predict = feats_pred.drop(["Driver"], axis=1)

    if model_training_features:
        X_predict = preprocess_for_prediction(X_predict, model_training_features)
    else:
        print(
            "WARNING: Proceeding without explicit feature alignment as model_training_features list was not available."
        )
        # Basic check: if model has feature_names_in_ (sklearn >= 0.24)
        if hasattr(model, "feature_names_in_"):
            try:
                X_predict = X_predict[model.feature_names_in_]
            except KeyError as e_key:
                print(
                    f"Feature mismatch: Model expects features {model.feature_names_in_}, but got {X_predict.columns}. Error: {e_key}"
                )
                print(
                    "Please ensure engineer_features() and preprocessing create the correct features."
                )
                return pd.DataFrame()
        else:  # Older sklearn or different model type
            print(
                "Model does not have feature_names_in_. Cannot guarantee feature order. Predictions may fail or be inaccurate."
            )

    # --- Make Predictions ---
    try:
        raw_preds = model.predict(X_predict)
    except ValueError as ve:
        print(f"ValueError during prediction: {ve}")
        print(
            "This often means the features in X_predict do not match what the model was trained on."
        )
        print(
            f"Features expected by model (if available): {getattr(model, 'feature_names_in_', 'N/A')}"
        )
        print(f"Features provided for prediction: {X_predict.columns.tolist()}")
        return pd.DataFrame()

    # Combine drivers with their raw scores
    predictions_df = pd.DataFrame(
        {"Driver": drivers_for_prediction, "predicted_raw_score": raw_preds}
    )

    # Assign unique ranks. Lower score = better predicted position.
    predictions_df["predicted_position"] = (
        predictions_df["predicted_raw_score"].rank(method="first").astype(int)
    )

    return predictions_df[["Driver", "predicted_position"]].sort_values(
        "predicted_position"
    )


def get_actual_results(year: int, gp_identifier: any) -> pd.DataFrame:
    """
    Fetch actual race results via FastF1.
    'gp_identifier' can be the event name or round number.
    """
    # setup_fastf1("cache") # Or fastf1.Cache.enable_cache("cache")
    fastf1.Cache.enable_cache("cache")

    print(f"Fetching actual Race results for {gp_identifier}, {year}...")
    try:
        sess = fastf1.get_session(year, gp_identifier, "R")
        sess.load(
            laps=False, telemetry=False, weather=False, messages=False
        )  # Load only what's needed for results
    except Exception as e:
        print(
            f"Error fetching race session for actual results {gp_identifier}, {year}: {e}"
        )
        return pd.DataFrame()

    if sess.results is None or sess.results.empty:
        print(f"No actual race results found for {gp_identifier}, {year}.")
        return pd.DataFrame()

    if (
        "Abbreviation" not in sess.results.columns
        or "Position" not in sess.results.columns
    ):
        print(
            f"Warning: 'Abbreviation' or 'Position' missing in actual race results for {gp_identifier}, {year}."
        )
        return pd.DataFrame()

    results = sess.results[["Abbreviation", "Position"]].copy()
    results.rename(
        columns={"Abbreviation": "Driver", "Position": "actual_position"}, inplace=True
    )
    results["actual_position"] = pd.to_numeric(
        results["actual_position"], errors="coerce"
    )
    results.dropna(subset=["actual_position"], inplace=True)  # Remove DNFs etc.
    if not results.empty:
        results["actual_position"] = results["actual_position"].astype(int)
    return results


if __name__ == "__main__":
    # --- Configuration ---
    # Determine the upcoming weekend's race.
    # For this example, let's manually set it.
    # In a real scenario, you might fetch the schedule and find the next event.
    predict_year = 2025  # Assuming today is before Emilia Romagna 2025
    predict_gp_name = "Qatar Grand Prix"  # Use the official event name from fastf1.get_event_schedule()
    # Or use round number if known and more reliable:
    # predict_gp_identifier = 7 # Example: If Emilia Romagna is Round 7 in 2025

    # To find the correct identifier for "Emilia Romagna":
    # import fastf1
    # schedule_2025 = fastf1.get_event_schedule(2025)
    # print(schedule_2025[schedule_2025['EventName'].str.contains("Emilia Romagna", case=False)])
    # From that, you can get the RoundNumber or confirm EventName.
    # Let's assume we found Round 7 is "Emilia Romagna Grand Prix"
    predict_gp_identifier = predict_gp_name  # Or the round number e.g. 7

    model_file_path = "f1_race_model.pkl"

    print(f"--- F1 Prediction for {predict_gp_identifier} {predict_year} ---")

    # --- Stage 1: Make Predictions (Run this after Qualifying) ---
    print("\nSTAGE 1: Generating Predictions (based on Qualifying data)")
    predicted_positions = predict_race(
        predict_year, predict_gp_identifier, model_path=model_file_path
    )

    if not predicted_positions.empty:
        print("\nPredicted Finishing Positions:")
        print(predicted_positions)  # Already sorted by predict_race
    else:
        print("\nCould not generate predictions. Please check logs.")

    # --- Stage 2: Compare with Actual Results (Run this AFTER the Race) ---
    print("\n\nSTAGE 2: Comparing with Actual Race Results")
    print(
        "(This part will only work meaningfully AFTER the race has concluded and results are available)"
    )

    actual_race_results = get_actual_results(predict_year, predict_gp_identifier)

    if not predicted_positions.empty and not actual_race_results.empty:
        print("\nActual Race Results:")
        print(actual_race_results.sort_values("actual_position"))

        comparison_df = predicted_positions.merge(
            actual_race_results,
            on="Driver",
            how="inner",  # Inner join on common drivers
        )

        if not comparison_df.empty:
            comparison_df["error"] = (
                comparison_df["predicted_position"] - comparison_df["actual_position"]
            )
            print("\nComparison (Predicted vs. Actual):")
            # Sort by predicted position for consistency, or actual_position if preferred
            print(comparison_df.sort_values("predicted_position"))

            mae = comparison_df["error"].abs().mean()
            print(
                f"\nMean Absolute Error for {predict_gp_identifier} {predict_year}: {mae:.2f} positions"
            )
        else:
            print(
                "\nCould not merge predictions with actual results. (No common drivers or one set empty)."
            )
    elif predicted_positions.empty:
        print("\nCannot compare because predictions were not generated.")
    else:  # actual_race_results is empty but predictions were generated
        print(
            "\nActual race results not found or empty. Cannot perform comparison yet."
        )
