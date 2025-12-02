# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_absolute_error
# # import joblib

# # from f1_data import setup_fastf1, fetch_session_data
# # from f1_features import engineer_features


# # def prepare_dataset(year: int, gp: str):
# #     setup_fastf1("cache")
# #     laps = fetch_session_data(year, gp, "R")
# #     feats = engineer_features(laps)
# #     # Placeholder target: replace with real results pulled from session.results
# #     feats["position"] = feats.index + 1
# #     return feats


# # def train_model(df: pd.DataFrame, model_path: str = "f1_race_model.pkl"):
# #     X = df.drop(["Driver", "position"], axis=1)
# #     y = df["position"]
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.2, random_state=42
# #     )
# #     model = RandomForestRegressor(n_estimators=100, random_state=42)
# #     model.fit(X_train, y_train)
# #     preds = model.predict(X_test)
# #     print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
# #     joblib.dump(model, model_path)
# #     print(f"Model saved to {model_path}")
# #     return model


# # if __name__ == "__main__":
# #     df = prepare_dataset(2024, "Monza")
# #     train_model(df)
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# import joblib
# import fastf1  # Make sure this is here for prepare_dataset

# from f1_data import setup_fastf1, fetch_session_data
# from f1_features import engineer_features

# # ... (your working prepare_dataset function definition should be here) ...


# # Ensure this function definition is exactly as follows and at the correct indentation level:
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
#     # *** MINOR BUG FIX (see below) ***
#     joblib.dump(model, model_path)  # Corrected from model.model_path
#     print(f"Model saved to {model_path}")
#     return model


# def prepare_dataset(year: int, gp: str):
#     setup_fastf1("cache")  # Ensures cache is set up

#     # Fetch lap data that will be used to engineer features.
#     # These features are what you'd have *before* the race outcome is known.
#     # For race prediction, you might train on features from:
#     #   - The race itself (to see how lap data correlates to finishing position)
#     #   - Or Qualifying/Practice sessions from that GP weekend (if predicting race outcome from pre-race data)
#     # Your current f1_predict.py uses Qualifying data to predict the Race.
#     # So, for consistency in training, you should also use Qualifying data to generate features
#     # and then map those to the *actual Race* results.

#     print(f"Fetching Qualifying data for feature engineering for {gp} {year}...")
#     qualifying_laps = fetch_session_data(
#         year, gp, "Q"
#     )  # Use 'Q' for features if predicting based on Q
#     if qualifying_laps.empty:
#         print(f"No qualifying data found for {gp} {year}. Skipping.")
#         return pd.DataFrame()
#     feats = engineer_features(qualifying_laps)

#     try:
#         q_session = fastf1.get_session(year, gp, "Q")
#         # Load minimal data needed for results
#         q_session.load(laps=False, telemetry=False, weather=False, messages=False)
#         q_results = q_session.results

#         if q_results is not None and not q_results.empty:
#             q_positions = q_results[["Abbreviation", "Position"]].copy()
#             q_positions.rename(
#                 columns={"Abbreviation": "Driver", "Position": "QualifyingPosition"},
#                 inplace=True,
#             )
#             q_positions["QualifyingPosition"] = pd.to_numeric(
#                 q_positions["QualifyingPosition"], errors="coerce"
#             )

#             # Merge with lap-based features
#             feats = feats.merge(q_positions, on="Driver", how="left")
#             # Fill NA for QualifyingPosition (e.g., if a driver set lap times but had no official Q rank)
#             feats["QualifyingPosition"].fillna(
#                 20, inplace=True
#             )  # Or a higher number like 24, or median/mean
#         else:
#             print(
#                 f"Warning: Could not load qualifying results for {year} {gp} to get QualifyingPosition. Using default."
#             )
#             feats = feats.copy()  # Important to copy if modifying
#             feats["QualifyingPosition"] = 20  # Default value
#     except Exception as e:
#         print(
#             f"Error loading qualifying session results for {year} {gp}: {e}. Using default QualifyingPosition."
#         )
#         feats = feats.copy()  # Important to copy if modifying
#         feats["QualifyingPosition"] = 20  # Default value

#     # Fetch actual race results to use as the target
#     print(f"Fetching actual Race results for target variable for {gp} {year}...")
#     try:
#         race_session = fastf1.get_session(year, gp, "R")  # Get the Race session object

#         # CORRECTED LOAD CALL:
#         # Disable other data types if you primarily need 'session.results'.
#         # The .results attribute will be available after a successful .load().
#         race_session.load(laps=False, telemetry=False, weather=False, messages=False)
#         # If you find 'session.results' is empty or None after this,
#         # you might need to simply call race_session.load() with default arguments
#         # or ensure the session type ('R' for Race) actually has results data available.

#     except Exception as e:
#         print(f"Could not load race session for {year} {gp}: {e}")
#         return pd.DataFrame()

#     actual_results = race_session.results  # Access the .results attribute AFTER loading

#     if actual_results is None or actual_results.empty:
#         print(
#             f"Warning: Could not load actual race results (actual_results is None or empty) for {year} {gp} to create target variable."
#         )
#         return pd.DataFrame()

#     # Select Driver (usually Abbreviation) and Position, then rename for merging
#     # Driver identifiers in `laps` (from fetch_session_data -> sess.laps) is usually 'Driver' (abbreviation)
#     # Driver identifiers in `results` is usually 'Abbreviation'
#     results_subset = actual_results[["Abbreviation", "Position"]].copy()
#     results_subset.rename(
#         columns={"Abbreviation": "Driver", "Position": "position"}, inplace=True
#     )
#     results_subset["position"] = pd.to_numeric(
#         results_subset["position"], errors="coerce"
#     )  # Convert to numeric, DNF/DSQ etc. become NaN

#     # Merge actual positions into the features DataFrame
#     feats_with_target = feats.merge(results_subset, on="Driver", how="left")

#     # Drop rows where position is NaN (e.g., drivers who did not finish or were not classified in the race)
#     # Also drop rows if a driver from Qualifying did not participate/get classified in the Race
#     feats_with_target.dropna(subset=["position"], inplace=True)
#     if feats_with_target.empty:
#         print(
#             f"No data left after merging features with race results for {gp} {year}. Possible driver mismatch or no classified finishers."
#         )
#         return pd.DataFrame()

#     feats_with_target["position"] = feats_with_target["position"].astype(int)

#     print(f"Prepared dataset for {gp} {year} with {len(feats_with_target)} drivers.")
#     return feats_with_target


# if __name__ == "__main__":
#     # It's better to train on multiple races for a more robust model
#     # Example: Train on a few races from 2023 or 2024
#     training_data_frames = []
#     # IMPORTANT: As of May 2025, the 2024 season is complete.
#     # The 2025 season has just started.
#     # Let's assume you want to train on 2024 data.
#     races_2024 = {
#         "Bahrain": "Bahrain Grand Prix",  # Example: official name vs common name
#         "Jeddah": "Saudi Arabian Grand Prix",
#         "Melbourne": "Australian Grand Prix",
#         "Suzuka": "Japanese Grand Prix",
#         "Shanghai": "Chinese Grand Prix",
#         "Miami": "Miami Grand Prix",  # The actual Miami GP from last year
#         "Imola": "Emilia Romagna Grand Prix",
#         "Monaco": "Monaco Grand Prix",
#         "Montreal": "Canadian Grand Prix",
#         "Barcelona": "Spanish Grand Prix",
#         "Spielberg": "Austrian Grand Prix",  # Spielberg / Red Bull Ring
#         "Silverstone": "British Grand Prix",
#         "Budapest": "Hungarian Grand Prix",
#         "Spa": "Belgian Grand Prix",  # Spa-Francorchamps
#         "Zandvoort": "Dutch Grand Prix",
#         "Monza": "Italian Grand Prix",
#         "Baku": "Azerbaijan Grand Prix",
#         "Singapore": "Singapore Grand Prix",
#         "Austin": "United States Grand Prix",  # COTA
#         "Mexico City": "Mexico City Grand Prix",
#         "Sao Paulo": "Sao Paulo Grand Prix",  # Interlagos / Brazilian GP
#         "Las Vegas": "Las Vegas Grand Prix",
#         "Lusail": "Qatar Grand Prix",
#         "Yas Marina": "Abu Dhabi Grand Prix",
#     }

#     # Using a few diverse tracks from 2024 for training:
#     # Ensure FastF1 can resolve these names for `gp` parameter.
#     # Often, the city name or a common name works.
#     # If `fastf1.get_session(year, gp, session)` fails, you may need the exact event name.
#     # You can check `fastf1.get_event_schedule(2024)` to see valid event names.
#     # For `get_session`, the `gp` argument is usually the 'EventName' or 'RoundNumber'.
#     # Let's use event numbers for reliability for a small sample.
#     # Check schedule: fastf1.get_event_schedule(2024)
#     # For example:
#     # Round 1: Bahrain Grand Prix
#     # Round 6: Miami Grand Prix
#     # Round 14: Italian Grand Prix (Monza)

#     sample_gps_2024_by_round = [1, 6, 14]  # Bahrain, Miami, Monza by round number

#     for round_num in sample_gps_2024_by_round:
#         print(f"\n--- Preparing data for 2024 Round {round_num} ---")
#         df_gp = prepare_dataset(
#             2024, round_num
#         )  # Pass round number or exact event name
#         if not df_gp.empty:
#             training_data_frames.append(df_gp)

#     if not training_data_frames:
#         print("No training data could be prepared. Exiting training.")
#     else:
#         full_training_df = pd.concat(training_data_frames, ignore_index=True)
#         print(f"\n--- Training model on {len(full_training_df)} total entries ---")
#         print(full_training_df.head())
#         print(full_training_df.info())
#         # Ensure 'position' column exists and is not all NaN before training
#         if (
#             "position" in full_training_df.columns
#             and not full_training_df["position"].isnull().all()
#         ):
#             train_model(
#                 full_training_df
#             )  # train_model is already defined in your f1_train.py
#         else:
#             print(
#                 "Critical error: Target 'position' is missing or all NaN in the final training dataframe."
#             )

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import fastf1  # For fastf1.get_session and fastf1.get_event_schedule
import datetime  # For checking current date

# Assuming these are in your project structure
from f1_data import setup_fastf1, fetch_session_data
from f1_features import engineer_features

import numpy as np

# Disable pandas SettingWithCopyWarning for cleaner output during merges/slicing
# You can enable it by setting to 'warn' if you prefer for debugging.
pd.options.mode.chained_assignment = None


def prepare_dataset(year: int, gp_identifier: int | str) -> pd.DataFrame:
    """
    Prepares a dataset for a single Grand Prix event.
    Features are engineered from Qualifying lap data.
    Target 'position' is the actual Race finishing position.
    'QualifyingPosition' is also added as a feature.
    """
    # Setup FastF1 cache. It's okay to call this multiple times.
    # fastf1.Cache.enable_cache("cache") is the more direct FastF1 API call.
    setup_fastf1("cache")

    # 1. Fetch Qualifying data for feature engineering
    print(
        f"Fetching Qualifying data for feature engineering: {year}, GP ID: {gp_identifier}..."
    )
    try:
        # Assuming fetch_session_data returns laps from the 'Q' session
        qualifying_laps = fetch_session_data(year, gp_identifier, "Q")
        if qualifying_laps.empty:
            print(
                f"No qualifying lap data found for {year}, GP ID: {gp_identifier}. Skipping this event."
            )
            return pd.DataFrame()
    except Exception as e:
        print(
            f"Error fetching qualifying lap data for {year}, GP ID: {gp_identifier}: {e}. Skipping."
        )
        return pd.DataFrame()

    # 2. Engineer features from Qualifying laps
    # engineer_features should return a DataFrame with 'Driver' and other numeric features
    feats = engineer_features(qualifying_laps)
    if feats.empty:
        print(
            f"Feature engineering yielded no data for {year}, GP ID: {gp_identifier}. Skipping."
        )
        return pd.DataFrame()

    # 3. Add Qualifying Position as a feature
    print(
        f"Fetching Qualifying session results for 'QualifyingPosition': {year}, GP ID: {gp_identifier}..."
    )
    try:
        q_session = fastf1.get_session(year, gp_identifier, "Q")
        q_session.load(
            laps=False, telemetry=False, weather=False, messages=False
        )  # Only results needed
        q_results = q_session.results

        if q_results is not None and not q_results.empty:
            # Ensure 'Abbreviation' and 'Position' columns exist
            if (
                "Abbreviation" not in q_results.columns
                or "Position" not in q_results.columns
            ):
                print(
                    f"Warning: 'Abbreviation' or 'Position' missing in Q results for {year}, GP ID: {gp_identifier}. Defaulting QualifyingPosition."
                )
                feats = feats.copy()  # Ensure we're working with a copy
                feats["QualifyingPosition"] = 25  # Default high value
            else:
                q_positions = q_results[["Abbreviation", "Position"]].copy()
                q_positions.rename(
                    columns={
                        "Abbreviation": "Driver",
                        "Position": "QualifyingPosition",
                    },
                    inplace=True,
                )
                q_positions["QualifyingPosition"] = pd.to_numeric(
                    q_positions["QualifyingPosition"], errors="coerce"
                )
                feats = feats.merge(q_positions, on="Driver", how="left")
        else:
            print(
                f"Warning: Could not load qualifying results for {year}, GP ID: {gp_identifier}. Defaulting QualifyingPosition."
            )
            feats = feats.copy()
            feats["QualifyingPosition"] = 25  # Default high value

    except Exception as e:
        print(
            f"Error loading qualifying session results for {year}, GP ID: {gp_identifier}: {e}. Defaulting QualifyingPosition."
        )
        feats = feats.copy()
        feats["QualifyingPosition"] = 25  # Default high value

    # Fill NA for QualifyingPosition (e.g., if a driver set lap times but had no official Q rank, or Q results error)
    # Using a value like 25 (assuming max ~20-22 drivers, so this is "worse than last")
    feats["QualifyingPosition"].fillna(25, inplace=True)

    # 4. Fetch actual Race results for the target variable 'position'
    print(
        f"Fetching Race results for target 'position': {year}, GP ID: {gp_identifier}..."
    )
    try:
        race_session = fastf1.get_session(year, gp_identifier, "R")
        race_session.load(
            laps=False, telemetry=False, weather=False, messages=False
        )  # Only results needed
        actual_results = race_session.results
    except Exception as e:
        print(
            f"Could not load race session for results for {year}, GP ID: {gp_identifier}: {e}. Skipping event."
        )
        return pd.DataFrame()

    if actual_results is None or actual_results.empty:
        print(
            f"Warning: No actual race results found for {year}, GP ID: {gp_identifier}. Skipping event."
        )
        return pd.DataFrame()

    # Ensure 'Abbreviation' and 'Position' columns exist in race results
    if (
        "Abbreviation" not in actual_results.columns
        or "Position" not in actual_results.columns
    ):
        print(
            f"Warning: 'Abbreviation' or 'Position' missing in Race results for {year}, GP ID: {gp_identifier}. Skipping event."
        )
        return pd.DataFrame()

    results_subset = actual_results[["Abbreviation", "Position"]].copy()
    results_subset.rename(
        columns={"Abbreviation": "Driver", "Position": "position"}, inplace=True
    )
    results_subset["position"] = pd.to_numeric(
        results_subset["position"], errors="coerce"
    )

    # 5. Merge actual race positions into the features DataFrame
    # 'feats' should have a 'Driver' column from engineer_features
    if "Driver" not in feats.columns:
        print(
            f"CRITICAL: 'Driver' column missing from engineered features for {year}, GP ID: {gp_identifier}. Cannot merge target. Skipping."
        )
        return pd.DataFrame()

    feats_with_target = feats.merge(results_subset, on="Driver", how="left")

    # Drop rows where 'position' is NaN (e.g., drivers who DNF'd or features for drivers not in race results)
    feats_with_target.dropna(subset=["position"], inplace=True)
    if feats_with_target.empty:
        print(
            f"No data left after merging features with race results for {year}, GP ID: {gp_identifier}. Possible driver mismatch or no classified finishers."
        )
        return pd.DataFrame()

    feats_with_target["position"] = feats_with_target["position"].astype(int)

    print(
        f"Prepared dataset for {year}, GP ID: {gp_identifier} with {len(feats_with_target)} drivers."
    )
    return feats_with_target


# def train_model(df: pd.DataFrame, model_path: str = "f1_race_model.pkl"):
#     """
#     Trains a RandomForestRegressor model and saves it.
#     Handles basic one-hot encoding for any categorical features.
#     """
#     if "Driver" not in df.columns or "position" not in df.columns:
#         print("Error: DataFrame must contain 'Driver' and 'position' columns.")
#         return None

#     # Features (X) are all columns except 'Driver' and 'position'
#     X = df.drop(["Driver", "position"], axis=1)
#     y = df["position"]

#     if X.empty:
#         print(
#             "Error: No feature columns found after dropping 'Driver' and 'position'. Cannot train model."
#         )
#         return None

#     # --- Preprocessing: Handle categorical features with one-hot encoding ---
#     # Identify object columns (potential categorical features)
#     categorical_cols = X.select_dtypes(include=["object", "category"]).columns
#     if not categorical_cols.empty:
#         print(
#             f"Found categorical columns: {list(categorical_cols)}. Applying one-hot encoding..."
#         )
#         X = pd.get_dummies(
#             X, columns=categorical_cols, dummy_na=False
#         )  # dummy_na=False to avoid creating NaN columns

#     # --- Imputation for remaining NaNs in numeric columns (if any) ---
#     # RandomForestRegressor cannot handle NaNs. Impute with median.

#     numeric_cols_with_na = X.select_dtypes(include=np.number).isnull().any()
#     cols_to_impute = numeric_cols_with_na[numeric_cols_with_na].index
#     if not cols_to_impute.empty:
#         print(
#             f"Found NaNs in numeric columns: {list(cols_to_impute)}. Imputing with median..."
#         )
#         for col in cols_to_impute:
#             median_val = X[col].median()
#             X[col].fillna(median_val, inplace=True)
#             print(f"Imputed NaNs in '{col}' with median: {median_val}")

#     # --- Model Training ---
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = RandomForestRegressor(
#         n_estimators=100, random_state=42, n_jobs=-1
#     )  # n_jobs=-1 uses all processors
#     print("Training model...")
#     model.fit(X_train, y_train)

#     preds = model.predict(X_test)
#     mae = mean_absolute_error(y_test, preds)
#     print(f"Model MAE on test set: {mae:.2f}")

#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")
#     return model


if __name__ == "__main__":
    training_data_frames = []
    # Determine current year and the year before
    # As of May 2024, this would mean 2023 and 2024.
    # For May 2025, this would mean 2024 and 2025.
    today = datetime.date.today()
    current_year = today.year
    previous_year = current_year - 1
    years_to_process = [
        previous_year,
        current_year,
    ]  # e.g., [2023, 2024] or [2024, 2025]

    # Setup FastF1 cache globally once (though setup_fastf1 in prepare_dataset is also fine)
    # fastf1.Cache.enable_cache("cache")
    # setup_fastf1("cache") # If your function handles this well

    for year in years_to_process:
        print(f"\n{'='*10} Processing year: {year} {'='*10}")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"Could not fetch event schedule for {year}: {e}")
            continue

        # Filter for events that are likely Grand Prix races
        # EventName often contains "Grand Prix". RoundNumber is a good stable identifier.
        # EventFormat can be 'conventional', 'sprint_shootout', 'sprint'.
        # We are interested in the main Grand Prix event of a weekend.
        # Using RoundNumber to get session should resolve to the correct GP.

        processed_event_names = (
            set()
        )  # To avoid processing an event twice if schedule has quirks

        for _, event in schedule.iterrows():
            event_date = pd.to_datetime(event["EventDate"]).date()
            event_name = event["EventName"]
            round_number = event["RoundNumber"]

            # Skip future events for the current year
            if year == current_year and event_date >= today:
                print(
                    f"Skipping future event: {event_name} ({year}, Round {round_number}) scheduled for {event_date}"
                )
                continue

            # Skip if already processed (e.g., if schedule has duplicate-like entries)
            # Using a combination of year and round_number as a unique key for an event
            event_key = (year, round_number)
            if event_key in processed_event_names:
                continue

            # Heuristic to identify actual Grand Prix events (not pre-season tests etc.)
            # Most reliable is trying to fetch 'R' session.
            # Checking for "Grand Prix" in the name is a good filter.
            if "grand prix" not in event_name.lower():
                print(
                    f"Skipping event (does not appear to be a Grand Prix): {event_name} ({year}, Round {round_number})"
                )
                continue

            print(
                f"\n--- Preparing data for: {event_name} ({year}, Round {round_number}) ---"
            )

            # Use RoundNumber as gp_identifier, as it's robust for fastf1.get_session
            df_gp = prepare_dataset(year, round_number)

            if df_gp is not None and not df_gp.empty:
                training_data_frames.append(df_gp)
                processed_event_names.add(event_key)
            else:
                print(
                    f"No data prepared for {event_name} ({year}, Round {round_number})."
                )

    if not training_data_frames:
        print("\nNo training data could be prepared from any events. Exiting.")
    else:
        print(
            f"\n{'='*10} Consolidating data from {len(training_data_frames)} GPs {'='*10}"
        )
        full_training_df = pd.concat(training_data_frames, ignore_index=True)

        if full_training_df.empty:
            print("Consolidated training DataFrame is empty. Exiting.")
        else:
            print(f"Total training entries: {len(full_training_df)}")
            print("Sample of combined data (head):")
            print(full_training_df.head())
            print("\nInfo of combined data:")
            full_training_df.info()

            # Ensure 'position' column exists and is not all NaN before training
            if "position" not in full_training_df.columns:
                print(
                    "Critical error: Target 'position' column is missing in the final training dataframe."
                )
            elif full_training_df["position"].isnull().all():
                print(
                    "Critical error: Target 'position' is all NaN in the final training dataframe."
                )
            else:
                print(f"\n{'='*10} Starting Model Training {'='*10}")
                train_model(full_training_df)  # Default model path "f1_race_model.pkl"

    # Re-enable pandas warning if it was changed globally for other parts of a larger application
    # pd.options.mode.chained_assignment = 'warn'
