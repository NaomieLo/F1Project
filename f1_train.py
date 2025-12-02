# f1_train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import fastf1
import datetime
import numpy as np

# Assuming these are in your project structure
from f1_data import setup_fastf1, fetch_session_data
from f1_features import engineer_features, engineer_features_pre_event  # Import both

# Disable pandas SettingWithCopyWarning for cleaner output
pd.options.mode.chained_assignment = None


def prepare_dataset_for_training(
    year: int, gp_identifier: any, mode: str = "post_qualifying"
) -> pd.DataFrame:
    """
    Prepares a dataset for a single Grand Prix event for training.
    Features are engineered based on the specified mode:
    - 'post_qualifying': Uses features from the event's Qualifying session, including Q lap data
                         and actual QualifyingPosition.
    - 'pre_qualifying': Uses features generated from historical data *before* this event's
                        on-track sessions.

    Target 'position' is always the actual Race finishing position for that event.
    """
    setup_fastf1("cache")  # Ensures FastF1 cache is enabled
    print(f"\n--- Preparing dataset for: {year} GP {gp_identifier} (Mode: {mode}) ---")

    # --- Feature Engineering ---
    feats = pd.DataFrame()

    if mode == "post_qualifying":
        # 1a. Fetch Qualifying lap data for post-Q features
        print("Fetching Qualifying lap data for feature engineering...")
        qualifying_laps = fetch_session_data(year, gp_identifier, "Q")
        if qualifying_laps.empty:
            print(
                f"No Qualifying lap data found for {year} GP {gp_identifier}. Skipping for post-Q model."
            )
            return pd.DataFrame()

        # 1b. Engineer features from Q laps using engineer_features()
        # This function should return a DataFrame with 'Driver' and Q-derived features.
        feats_from_q_laps = engineer_features(qualifying_laps)
        if feats_from_q_laps.empty:
            print("Post-Q feature engineering from Q laps failed. Skipping.")
            return pd.DataFrame()

        # 1c. Add actual QualifyingPosition as a feature from the Q session results
        print("Fetching Qualifying session results for 'QualifyingPosition'...")
        try:
            q_session = fastf1.get_session(year, gp_identifier, "Q")
            q_session.load(
                laps=False, telemetry=False, weather=False, messages=False
            )  # Only need results
            q_results = q_session.results

            if (
                q_results is not None
                and not q_results.empty
                and "Abbreviation" in q_results.columns
                and "Position" in q_results.columns
            ):
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

                # Merge Q positions with features from Q laps
                feats = feats_from_q_laps.merge(q_positions, on="Driver", how="left")
                # Fill NaNs for QualifyingPosition (e.g. if a driver in Q laps has no official Q rank)
                feats.loc[:, "QualifyingPosition"] = feats["QualifyingPosition"].fillna(
                    25
                )  # Default (e.g. worse than last)
            else:
                print(
                    "Warning: Q results not available or incomplete for 'QualifyingPosition'. Adding default column."
                )
                feats = feats_from_q_laps.copy()  # Work with a copy
                feats["QualifyingPosition"] = (
                    25  # Add default QualifyingPosition column
                )
        except Exception as e:
            print(
                f"Error getting Q results for QualifyingPosition: {e}. Adding default column."
            )
            feats = feats_from_q_laps.copy()  # Ensure feats exists and is a copy
            feats["QualifyingPosition"] = 25

        # (Optional for post_qualifying mode): You could also merge pre-event features here
        # if your post_qualifying model benefits from historical context + current Q data.
        # print("Fetching pre-event features to combine with post-Q features...")
        # if 'Driver' in feats.columns and not feats.empty:
        #    drivers_for_pre_event_context = feats['Driver'].unique().tolist()
        #    pre_event_context_feats = engineer_features_pre_event(year, gp_identifier, drivers_for_pre_event_context)
        #    if not pre_event_context_feats.empty:
        #        feats = feats.merge(pre_event_context_feats, on="Driver", how="left", suffixes=('', '_hist'))
        #    else:
        #        print("Could not generate pre-event context features to merge.")
        # else:
        #    print("No drivers found from Q features to get pre-event context.")

    elif mode == "pre_qualifying":
        # For pre-Q mode, we first need to know which drivers participated in the target race
        # to generate historical features *for them* leading up to *this specific race*.
        print(
            "Fetching Race session to identify participating drivers for pre-Q feature context..."
        )
        try:
            # Load race results to get the list of drivers who actually started/finished
            race_session_for_drivers = fastf1.get_session(year, gp_identifier, "R")
            race_session_for_drivers.load(
                laps=False, telemetry=False, weather=False, messages=False
            )

            if (
                race_session_for_drivers.results is None
                or race_session_for_drivers.results.empty
                or "Abbreviation" not in race_session_for_drivers.results.columns
            ):
                print(
                    f"No Race results or 'Abbreviation' column to get driver list for {year} GP {gp_identifier}. Skipping for pre-Q model."
                )
                return pd.DataFrame()

            participating_drivers = (
                race_session_for_drivers.results["Abbreviation"].unique().tolist()
            )
            if not participating_drivers:
                print(
                    "No participating drivers identified from race results. Skipping pre-Q."
                )
                return pd.DataFrame()

            print(
                f"Engineering pre-event features for {len(participating_drivers)} identified drivers..."
            )
            # `engineer_features_pre_event` uses data *before* the current `gp_identifier`
            feats = engineer_features_pre_event(
                year, gp_identifier, participating_drivers
            )

        except Exception as e:
            print(
                f"Error during pre-Q feature preparation (getting drivers or engineering features): {e}"
            )
            return pd.DataFrame()

    else:
        raise ValueError(
            f"Invalid mode specified for prepare_dataset_for_training: {mode}"
        )

    # If feature engineering (for either mode) failed or returned empty
    if feats.empty:
        print(
            f"Feature engineering returned no data for mode '{mode}'. Skipping event."
        )
        return pd.DataFrame()
    if "Driver" not in feats.columns:
        print(
            f"CRITICAL: 'Driver' column missing after feature engineering (Mode: {mode}). Skipping."
        )
        return pd.DataFrame()

    # --- Fetch Actual Race Results (Target Variable) ---
    # This is common for both modes, as the target is always the race outcome.
    print(f"Fetching actual Race results for target 'position'...")
    try:
        race_session_target = fastf1.get_session(year, gp_identifier, "R")
        race_session_target.load(
            laps=False, telemetry=False, weather=False, messages=False
        )
        actual_results = race_session_target.results

        if (
            actual_results is None
            or actual_results.empty
            or "Abbreviation" not in actual_results.columns
            or "Position" not in actual_results.columns
        ):
            print(
                f"No actual race results or required columns ('Abbreviation', 'Position') for target. Skipping."
            )
            return pd.DataFrame()
    except Exception as e:
        print(f"Could not load race session for target variable: {e}. Skipping.")
        return pd.DataFrame()

    results_subset = actual_results[["Abbreviation", "Position"]].copy()
    results_subset.rename(
        columns={"Abbreviation": "Driver", "Position": "position"}, inplace=True
    )
    results_subset["position"] = pd.to_numeric(
        results_subset["position"], errors="coerce"
    )

    # Merge features with the target variable
    feats_with_target = feats.merge(
        results_subset, on="Driver", how="left"
    )  # Left merge to keep all drivers with features

    # Drop rows where 'position' is NaN (e.g., drivers in Q/pre-Q features but DNF/DSQ in race without rank)
    # Or drivers for whom we couldn't get a race result.
    feats_with_target.dropna(subset=["position"], inplace=True)

    if feats_with_target.empty:
        print(
            "No data remaining after merging features with target or after dropping NaNs in 'position'. Skipping."
        )
        return pd.DataFrame()

    feats_with_target.loc[:, "position"] = feats_with_target["position"].astype(int)

    num_drivers = len(feats_with_target)
    print(
        f"Successfully prepared dataset for {year} GP {gp_identifier} (Mode: {mode}) with {num_drivers} drivers."
    )
    return feats_with_target


def train_and_save_model(df: pd.DataFrame, model_path: str):
    """
    Trains a RandomForestRegressor model using the provided DataFrame,
    handles preprocessing (OHE, imputation), and saves the model as a bundle
    (model object + list of feature names).
    """
    print(f"\n--- Training model to be saved at: {model_path} ---")
    if df.empty:
        print(f"Received empty DataFrame for training {model_path}. Skipping.")
        return None
    if "Driver" not in df.columns or "position" not in df.columns:
        print(
            f"DataFrame for {model_path} must contain 'Driver' and 'position' columns. Skipping."
        )
        return None

    X = df.drop(["Driver", "position"], axis=1)
    y = df["position"]

    if X.empty:
        print(
            f"No feature columns found after dropping 'Driver' and 'position' for {model_path}. Skipping."
        )
        return None

    print(f"Initial features for {model_path}: {X.columns.tolist()}")

    # --- Preprocessing: Handle categorical features with one-hot encoding ---
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if not categorical_cols.empty:
        print(f"Applying one-hot encoding to: {list(categorical_cols)}")
        X = pd.get_dummies(
            X, columns=categorical_cols, dummy_na=False, prefix=categorical_cols
        )

    # --- Imputation for remaining NaNs in numeric columns ---
    # Ensure all columns are numeric that should be, convert if necessary
    for col in X.columns:
        if (
            X[col].dtype == "object"
        ):  # Or other non-numeric types that should be numeric
            try:
                X.loc[:, col] = pd.to_numeric(X[col], errors="coerce")
                print(f"Converted column '{col}' to numeric.")
            except Exception as e_conv:
                print(
                    f"Could not convert column '{col}' to numeric: {e_conv}. It might be dropped or cause issues."
                )

    numeric_cols = X.select_dtypes(include=np.number).columns
    cols_to_impute = X[numeric_cols].isnull().any()
    cols_to_impute_list = cols_to_impute[cols_to_impute].index.tolist()

    if cols_to_impute_list:
        print(f"Imputing NaNs with median in numeric columns: {cols_to_impute_list}")
        for col in cols_to_impute_list:
            median_val = X[col].median()
            X.loc[:, col] = X[col].fillna(median_val)
            # print(f"Imputed NaNs in '{col}' with median: {median_val}")

    # Check for any remaining NaNs after imputation (should ideally be none in numeric features)
    if X.isnull().any().any():
        print(
            f"WARNING: NaNs still present in features for {model_path} after imputation. Columns: {X.columns[X.isnull().any()].tolist()}"
        )
        # Aggressive: drop rows with any NaNs left in X, or drop columns.
        # X.dropna(inplace=True)
        # y = y[X.index] # Align y if rows are dropped from X

    print(f"Final features for training {model_path}: {X.columns.tolist()}")

    # --- Model Training ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except ValueError as e_split:
        print(
            f"Error during train_test_split for {model_path} (possibly due to insufficient data after NaN handling): {e_split}"
        )
        print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
        return None

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print(f"Training RandomForestRegressor for {model_path}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model MAE on test set for {model_path}: {mae:.2f}")

    # Save model and feature list as a bundle
    model_training_features = X_train.columns.tolist()
    model_bundle = {"model": model, "features": model_training_features}

    joblib.dump(model_bundle, model_path)
    print(f"Model (and feature list) for {model_path} saved successfully.")
    return model_bundle


if __name__ == "__main__":
    # --- Configuration ---
    today = datetime.date.today()
    current_year = today.year
    # For robust training, go back at least one full previous season, plus completed races of current season
    # Example: if today is mid-2025, years_to_process = [2024, 2025]
    # If today is early 2025, years_to_process = [2023, 2024] (as 2025 might have no completed races)

    # Let's define years based on assumption we want full previous season and current
    # If current year has few races, might want to include year before previous_year too
    # For this example, let's use 2 full prior seasons if current year just started.

    # Simplified logic: use the last two fully completed years if current year is too early.
    # More robust: a list of specific years known to have good data.
    # years_to_process = [2023, 2024] # Manually set for now

    # Dynamic year selection
    if today.month < 3:  # If very early in the year, current year might have no data
        years_to_process = [current_year - 2, current_year - 1]
    else:
        years_to_process = [current_year - 1, current_year]

    print(f"Training data will be sourced from years: {years_to_process}")

    processed_events_log = (
        set()
    )  # To avoid processing an event multiple times if schedule is weird

    # --- Train POST-QUALIFYING Model ---
    model_post_q_path = "f1_model_post_qualifying.pkl"
    print(
        "\n"
        + "=" * 20
        + f" GATHERING DATA FOR POST-QUALIFYING MODEL ({model_post_q_path}) "
        + "=" * 20
    )
    training_dfs_post_q = []
    for year in years_to_process:
        print(f"Processing year {year} for POST-Q model data...")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e_sched:
            print(
                f"Could not fetch schedule for year {year}: {e_sched}. Skipping year."
            )
            continue

        for _, event in schedule.iterrows():
            event_key = (year, event["RoundNumber"], "post_q")
            if event_key in processed_events_log:
                continue

            event_date = pd.to_datetime(event["EventDate"]).date()
            # Skip future events of the most current year being processed
            if year == current_year and event_date >= today:
                # print(f"Skipping future event for post-Q: {event['EventName']} {year}")
                continue
            if "grand prix" not in event["EventName"].lower():  # Filter for actual GPs
                continue

            df_gp = prepare_dataset_for_training(
                year, event["RoundNumber"], mode="post_qualifying"
            )
            if df_gp is not None and not df_gp.empty:
                training_dfs_post_q.append(df_gp)
            processed_events_log.add(event_key)

    if training_dfs_post_q:
        full_df_post_q = pd.concat(training_dfs_post_q, ignore_index=True)
        if not full_df_post_q.empty:
            print(f"Total entries for POST-QUALIFYING model: {len(full_df_post_q)}")
            train_and_save_model(full_df_post_q, model_post_q_path)
        else:
            print("Concatenated DataFrame for POST-QUALIFYING model is empty.")
    else:
        print(
            "No dataframes were collected for POST-QUALIFYING model training. Model not trained."
        )

    # --- Train PRE-QUALIFYING Model ---
    model_pre_q_path = "f1_model_pre_qualifying.pkl"
    print(
        "\n"
        + "=" * 20
        + f" GATHERING DATA FOR PRE-QUALIFYING MODEL ({model_pre_q_path}) "
        + "=" * 20
    )
    training_dfs_pre_q = []
    processed_events_log.clear()  # Reset for pre-q mode if needed, or use different keys
    for year in years_to_process:
        print(f"Processing year {year} for PRE-Q model data...")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e_sched:
            print(
                f"Could not fetch schedule for year {year}: {e_sched}. Skipping year."
            )
            continue

        for _, event in schedule.iterrows():
            event_key = (year, event["RoundNumber"], "pre_q")
            if event_key in processed_events_log:
                continue

            event_date = pd.to_datetime(event["EventDate"]).date()
            if year == current_year and event_date >= today:
                # print(f"Skipping future event for pre-Q: {event['EventName']} {year}")
                continue
            if "grand prix" not in event["EventName"].lower():
                continue

            df_gp = prepare_dataset_for_training(
                year, event["RoundNumber"], mode="pre_qualifying"
            )
            if df_gp is not None and not df_gp.empty:
                training_dfs_pre_q.append(df_gp)
            processed_events_log.add(event_key)

    if training_dfs_pre_q:
        full_df_pre_q = pd.concat(training_dfs_pre_q, ignore_index=True)
        if not full_df_pre_q.empty:
            print(f"Total entries for PRE-QUALIFYING model: {len(full_df_pre_q)}")
            train_and_save_model(full_df_pre_q, model_pre_q_path)
        else:
            print("Concatenated DataFrame for PRE-QUALIFYING model is empty.")
    else:
        print(
            "No dataframes were collected for PRE-QUALIFYING model training. Model not trained."
        )

    print("\n--- Training Process Finished ---")
