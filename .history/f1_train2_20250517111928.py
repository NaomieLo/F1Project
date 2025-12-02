# In f1_train.py

# ... (imports: pandas, sklearn, joblib, fastf1, datetime, numpy)
# ... (from f1_data import setup_fastf1, fetch_session_data)
# ... (from f1_features import engineer_features, engineer_features_pre_event) # Assume both exist

pd.options.mode.chained_assignment = None


def prepare_dataset_for_training(
    year: int, gp_identifier: any, mode: str = "post_qualifying"
) -> pd.DataFrame:
    """
    Prepares dataset for training.
    mode: 'post_qualifying' or 'pre_qualifying'
    """
    setup_fastf1("cache")
    print(f"--- Preparing dataset for {year} GP {gp_identifier} (Mode: {mode}) ---")

    # --- Feature Engineering ---
    if mode == "post_qualifying":
        # Fetch Q laps for features
        print(f"Fetching Qualifying lap data for feature engineering...")
        qualifying_laps = fetch_session_data(year, gp_identifier, "Q")
        if qualifying_laps.empty:
            print(
                f"No Q laps found for {year} GP {gp_identifier}. Skipping for post-Q model."
            )
            return pd.DataFrame()

        # engineer_features now represents your post-Q feature engineering
        # It should take Q laps and produce features including those derived from Q.
        feats = engineer_features(
            qualifying_laps
        )  # This is your existing function for post-Q
        if feats.empty:
            print("Post-Q feature engineering failed. Skipping.")
            return pd.DataFrame()

        # Add QualifyingPosition from the Q session itself
        print(f"Fetching Qualifying session results for 'QualifyingPosition'...")
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
                feats["QualifyingPosition"].fillna(
                    25, inplace=True
                )  # Default for any NAs
            else:
                print(
                    "Warning: Q results not available or incomplete. Defaulting QualifyingPosition."
                )
                feats["QualifyingPosition"] = 25  # Add if not merged
        except Exception as e:
            print(f"Error getting Q results for QualifyingPosition: {e}. Defaulting.")
            if (
                "QualifyingPosition" not in feats.columns
            ):  # Check if column needs to be added
                feats = feats.copy()  # ensure it's a copy if adding a new column
                feats["QualifyingPosition"] = 25
            else:  # If column exists from merge but failed, still fill NAs
                feats["QualifyingPosition"].fillna(25, inplace=True)

    elif mode == "pre_qualifying":
        # For pre-Q, we need a list of drivers who participated in the race to generate historical features for.
        print(
            "Fetching Race session to get participating drivers for pre-Q feature context..."
        )
        try:
            race_session_for_drivers = fastf1.get_session(year, gp_identifier, "R")
            race_session_for_drivers.load(
                laps=False, telemetry=False, weather=False, messages=False, results=True
            )  # Need results for drivers
            if (
                race_session_for_drivers.results is None
                or race_session_for_drivers.results.empty
            ):
                print(
                    f"No race results to get driver list for {year} GP {gp_identifier}. Skipping for pre-Q model."
                )
                return pd.DataFrame()

            # Get driver abbreviations from race results
            # (Assuming 'Abbreviation' is the column with 'VER', 'HAM' etc.)
            if "Abbreviation" not in race_session_for_drivers.results.columns:
                print(
                    "Cannot find 'Abbreviation' in race results to list drivers. Skipping pre-Q."
                )
                return pd.DataFrame()

            participating_drivers = (
                race_session_for_drivers.results["Abbreviation"].unique().tolist()
            )
            if not participating_drivers:
                print(
                    "No participating drivers found from race results. Skipping pre-Q."
                )
                return pd.DataFrame()

            print(
                f"Engineering pre-event features for {len(participating_drivers)} drivers..."
            )
            # `engineer_features_pre_event` takes the event context (year, gp_id for which we're building history)
            # and the list of drivers who actually raced (for whom we need features).
            # It MUST NOT use any data from the current gp_identifier's Q or R sessions for feature values.
            feats = engineer_features_pre_event(
                year, gp_identifier, participating_drivers
            )
            if feats.empty:
                print("Pre-Q feature engineering failed. Skipping.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error during pre-Q feature prep: {e}")
            return pd.DataFrame()
    else:
        raise ValueError(f"Invalid mode for prepare_dataset_for_training: {mode}")

    if "Driver" not in feats.columns:
        print(
            f"CRITICAL: 'Driver' column missing after feature engineering (Mode: {mode}). Skipping."
        )
        return pd.DataFrame()

    # --- Fetch Actual Race Results (Target Variable) ---
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
            print(f"No actual race results/driver/position info for target. Skipping.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Could not load race session for target: {e}. Skipping.")
        return pd.DataFrame()

    results_subset = actual_results[["Abbreviation", "Position"]].copy()
    results_subset.rename(
        columns={"Abbreviation": "Driver", "Position": "position"}, inplace=True
    )
    results_subset["position"] = pd.to_numeric(
        results_subset["position"], errors="coerce"
    )

    feats_with_target = feats.merge(results_subset, on="Driver", how="left")
    feats_with_target.dropna(
        subset=["position"], inplace=True
    )  # Drop drivers with no race finish position

    if feats_with_target.empty:
        print("No data after merging features with target. Skipping.")
        return pd.DataFrame()

    feats_with_target["position"] = feats_with_target["position"].astype(int)
    print(
        f"Prepared dataset for {year} GP {gp_identifier} (Mode: {mode}) with {len(feats_with_target)} drivers."
    )
    return feats_with_target


def train_and_save_model(df: pd.DataFrame, model_path: str):
    # (This is your existing train_model function, slightly renamed for clarity)
    if (
        "Driver" not in df.columns or "position" not in df.columns
    ):  # ... (rest of the function remains same)
        print("Error: DataFrame must contain 'Driver' and 'position' columns.")
        return None
    X = df.drop(["Driver", "position"], axis=1)
    y = df["position"]
    if X.empty:  # ...
        print("Error: No feature columns found after dropping 'Driver' and 'position'.")
        return None

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)

    numeric_cols_with_na = X.select_dtypes(include=np.number).isnull().any()
    cols_to_impute = numeric_cols_with_na[numeric_cols_with_na].index
    if not cols_to_impute.empty:
        for col in cols_to_impute:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model MAE on test set for {model_path}: {mae:.2f}")

    model_training_features = X_train.columns.tolist()
    model_bundle = {"model": model, "features": model_training_features}
    joblib.dump(model_bundle, model_path)
    print(f"Model and feature list saved to {model_path}")
    return model_bundle


if __name__ == "__main__":
    today = datetime.date.today()
    current_year = today.year
    previous_year = current_year - 1
    years_to_process = [previous_year, current_year]

    # --- Train POST-QUALIFYING Model ---
    print("\n" + "=" * 20 + " TRAINING POST-QUALIFYING MODEL " + "=" * 20)
    training_dfs_post_q = []
    for year in years_to_process:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            event_date = pd.to_datetime(event["EventDate"]).date()
            if year == current_year and event_date >= today:
                continue
            if "grand prix" not in event["EventName"].lower():
                continue

            df_gp = prepare_dataset_for_training(
                year, event["RoundNumber"], mode="post_qualifying"
            )
            if not df_gp.empty:
                training_dfs_post_q.append(df_gp)

    if training_dfs_post_q:
        full_df_post_q = pd.concat(training_dfs_post_q, ignore_index=True)
        if not full_df_post_q.empty:
            train_and_save_model(full_df_post_q, "f1_model_post_qualifying.pkl")
        else:
            print("No data available for POST-QUALIFYING model training.")
    else:
        print("No dataframes collected for POST-QUALIFYING model training.")

    # --- Train PRE-QUALIFYING Model ---
    print("\n" + "=" * 20 + " TRAINING PRE-QUALIFYING MODEL " + "=" * 20)
    training_dfs_pre_q = []
    for year in years_to_process:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            event_date = pd.to_datetime(event["EventDate"]).date()
            if year == current_year and event_date >= today:
                continue
            if "grand prix" not in event["EventName"].lower():
                continue

            df_gp = prepare_dataset_for_training(
                year, event["RoundNumber"], mode="pre_qualifying"
            )
            if not df_gp.empty:
                training_dfs_pre_q.append(df_gp)

    if training_dfs_pre_q:
        full_df_pre_q = pd.concat(training_dfs_pre_q, ignore_index=True)
        if not full_df_pre_q.empty:
            train_and_save_model(full_df_pre_q, "f1_model_pre_qualifying.pkl")
        else:
            print("No data available for PRE-QUALIFYING model training.")
    else:
        print("No dataframes collected for PRE-QUALIFYING model training.")
