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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import fastf1  # Make sure this is here for prepare_dataset

from f1_data import setup_fastf1, fetch_session_data
from f1_features import engineer_features

# ... (your working prepare_dataset function definition should be here) ...


# Ensure this function definition is exactly as follows and at the correct indentation level:
def train_model(df: pd.DataFrame, model_path: str = "f1_race_model.pkl"):
    X = df.drop(["Driver", "position"], axis=1)
    y = df["position"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    # *** MINOR BUG FIX (see below) ***
    joblib.dump(model, model_path)  # Corrected from model.model_path
    print(f"Model saved to {model_path}")
    return model


def prepare_dataset(year: int, gp: str):
    setup_fastf1("cache")  # Ensures cache is set up

    # Fetch lap data that will be used to engineer features.
    # These features are what you'd have *before* the race outcome is known.
    # For race prediction, you might train on features from:
    #   - The race itself (to see how lap data correlates to finishing position)
    #   - Or Qualifying/Practice sessions from that GP weekend (if predicting race outcome from pre-race data)
    # Your current f1_predict.py uses Qualifying data to predict the Race.
    # So, for consistency in training, you should also use Qualifying data to generate features
    # and then map those to the *actual Race* results.

    print(f"Fetching Qualifying data for feature engineering for {gp} {year}...")
    qualifying_laps = fetch_session_data(
        year, gp, "Q"
    )  # Use 'Q' for features if predicting based on Q
    if qualifying_laps.empty:
        print(f"No qualifying data found for {gp} {year}. Skipping.")
        return pd.DataFrame()
    feats = engineer_features(qualifying_laps)

    # Fetch actual race results to use as the target
    print(f"Fetching actual Race results for target variable for {gp} {year}...")
    try:
        race_session = fastf1.get_session(year, gp, "R")  # Get the Race session object

        # CORRECTED LOAD CALL:
        # Disable other data types if you primarily need 'session.results'.
        # The .results attribute will be available after a successful .load().
        race_session.load(laps=False, telemetry=False, weather=False, messages=False)
        # If you find 'session.results' is empty or None after this,
        # you might need to simply call race_session.load() with default arguments
        # or ensure the session type ('R' for Race) actually has results data available.

    except Exception as e:
        print(f"Could not load race session for {year} {gp}: {e}")
        return pd.DataFrame()

    actual_results = race_session.results  # Access the .results attribute AFTER loading

    if actual_results is None or actual_results.empty:
        print(
            f"Warning: Could not load actual race results (actual_results is None or empty) for {year} {gp} to create target variable."
        )
        return pd.DataFrame()

    # Select Driver (usually Abbreviation) and Position, then rename for merging
    # Driver identifiers in `laps` (from fetch_session_data -> sess.laps) is usually 'Driver' (abbreviation)
    # Driver identifiers in `results` is usually 'Abbreviation'
    results_subset = actual_results[["Abbreviation", "Position"]].copy()
    results_subset.rename(
        columns={"Abbreviation": "Driver", "Position": "position"}, inplace=True
    )
    results_subset["position"] = pd.to_numeric(
        results_subset["position"], errors="coerce"
    )  # Convert to numeric, DNF/DSQ etc. become NaN

    # Merge actual positions into the features DataFrame
    feats_with_target = feats.merge(results_subset, on="Driver", how="left")

    # Drop rows where position is NaN (e.g., drivers who did not finish or were not classified in the race)
    # Also drop rows if a driver from Qualifying did not participate/get classified in the Race
    feats_with_target.dropna(subset=["position"], inplace=True)
    if feats_with_target.empty:
        print(
            f"No data left after merging features with race results for {gp} {year}. Possible driver mismatch or no classified finishers."
        )
        return pd.DataFrame()

    feats_with_target["position"] = feats_with_target["position"].astype(int)

    print(f"Prepared dataset for {gp} {year} with {len(feats_with_target)} drivers.")
    return feats_with_target


if __name__ == "__main__":
    # It's better to train on multiple races for a more robust model
    # Example: Train on a few races from 2023 or 2024
    training_data_frames = []
    # IMPORTANT: As of May 2025, the 2024 season is complete.
    # The 2025 season has just started.
    # Let's assume you want to train on 2024 data.
    races_2024 = {
        "Bahrain": "Bahrain Grand Prix",  # Example: official name vs common name
        "Jeddah": "Saudi Arabian Grand Prix",
        "Melbourne": "Australian Grand Prix",
        "Suzuka": "Japanese Grand Prix",
        "Shanghai": "Chinese Grand Prix",
        "Miami": "Miami Grand Prix",  # The actual Miami GP from last year
        "Imola": "Emilia Romagna Grand Prix",
        "Monaco": "Monaco Grand Prix",
        "Montreal": "Canadian Grand Prix",
        "Barcelona": "Spanish Grand Prix",
        "Spielberg": "Austrian Grand Prix",  # Spielberg / Red Bull Ring
        "Silverstone": "British Grand Prix",
        "Budapest": "Hungarian Grand Prix",
        "Spa": "Belgian Grand Prix",  # Spa-Francorchamps
        "Zandvoort": "Dutch Grand Prix",
        "Monza": "Italian Grand Prix",
        "Baku": "Azerbaijan Grand Prix",
        "Singapore": "Singapore Grand Prix",
        "Austin": "United States Grand Prix",  # COTA
        "Mexico City": "Mexico City Grand Prix",
        "Sao Paulo": "Sao Paulo Grand Prix",  # Interlagos / Brazilian GP
        "Las Vegas": "Las Vegas Grand Prix",
        "Lusail": "Qatar Grand Prix",
        "Yas Marina": "Abu Dhabi Grand Prix",
    }

    # Using a few diverse tracks from 2024 for training:
    # Ensure FastF1 can resolve these names for `gp` parameter.
    # Often, the city name or a common name works.
    # If `fastf1.get_session(year, gp, session)` fails, you may need the exact event name.
    # You can check `fastf1.get_event_schedule(2024)` to see valid event names.
    # For `get_session`, the `gp` argument is usually the 'EventName' or 'RoundNumber'.
    # Let's use event numbers for reliability for a small sample.
    # Check schedule: fastf1.get_event_schedule(2024)
    # For example:
    # Round 1: Bahrain Grand Prix
    # Round 6: Miami Grand Prix
    # Round 14: Italian Grand Prix (Monza)

    sample_gps_2024_by_round = [1, 6, 14]  # Bahrain, Miami, Monza by round number

    for round_num in sample_gps_2024_by_round:
        print(f"\n--- Preparing data for 2024 Round {round_num} ---")
        df_gp = prepare_dataset(
            2024, round_num
        )  # Pass round number or exact event name
        if not df_gp.empty:
            training_data_frames.append(df_gp)

    if not training_data_frames:
        print("No training data could be prepared. Exiting training.")
    else:
        full_training_df = pd.concat(training_data_frames, ignore_index=True)
        print(f"\n--- Training model on {len(full_training_df)} total entries ---")
        print(full_training_df.head())
        print(full_training_df.info())
        # Ensure 'position' column exists and is not all NaN before training
        if (
            "position" in full_training_df.columns
            and not full_training_df["position"].isnull().all()
        ):
            train_model(
                full_training_df
            )  # train_model is already defined in your f1_train.py
        else:
            print(
                "Critical error: Target 'position' is missing or all NaN in the final training dataframe."
            )
