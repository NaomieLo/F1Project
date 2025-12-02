# import pandas as pd


# def engineer_features(laps: pd.DataFrame) -> pd.DataFrame:
#     """
#     Aggregate per-driver features for ML:
#       - avg lap time (sec)
#       - lap time variance
#       - pit stop count
#       - average track temp and rainfall
#     """
#     # Convert LapTime timedeltas to numeric seconds
#     laps["lap_sec"] = laps["LapTime"].dt.total_seconds()

#     # Ensure weather columns exist and fill missing values
#     weather_cols = ["TrackTemp", "Rainfall", "WindSpeed", "WindDirection"]
#     for col in weather_cols:
#         if col not in laps.columns:
#             laps[col] = pd.NA
#     laps[weather_cols] = laps[weather_cols].ffill().bfill()

#     # Per-driver aggregations
#     avg_lap = laps.groupby("Driver")["lap_sec"].mean().rename("avg_lap_time")
#     std_lap = laps.groupby("Driver")["lap_sec"].std().rename("std_lap_time")
#     pit_count = (
#         laps[laps["PitOutTime"].notnull()].groupby("Driver").size().rename("pit_count")
#     )
#     avg_temp = laps.groupby("Driver")["TrackTemp"].mean().rename("avg_track_temp")
#     avg_rain = laps.groupby("Driver")["Rainfall"].mean().rename("avg_rainfall")

#     features = pd.concat(
#         [avg_lap, std_lap, pit_count, avg_temp, avg_rain], axis=1
#     ).fillna(0)
#     return features.reset_index()

import pandas as pd
import numpy as np
import fastf1
from fastf1.ergast import Ergast  # For fetching historical standings if needed


# Helper function to convert lap times to seconds if not already
def _to_seconds(lap_time_delta):
    if pd.isna(lap_time_delta):
        return np.nan
    return lap_time_delta.total_seconds()


def engineer_features(laps_q_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from qualifying lap data (POST-QUALIFYING features).
    Assumes laps_q_df is the DataFrame of laps from a Qualifying session.

    Returns:
        Pandas DataFrame with 'Driver' column (abbreviations) and other numeric features.
    """
    if laps_q_df.empty:
        print("engineer_features: Received empty qualifying laps DataFrame.")
        return pd.DataFrame()

    print(f"engineer_features: Processing {len(laps_q_df)} qualifying laps.")

    # Ensure 'Driver' column (abbreviations) exists
    if "Driver" not in laps_q_df.columns:
        print("engineer_features: 'Driver' column missing in Q laps. Cannot proceed.")
        return pd.DataFrame()

    # Convert LapTime to seconds for calculations
    laps_q_df["LapTimeSeconds"] = laps_q_df["LapTime"].apply(_to_seconds)

    # Filter out laps without time (e.g., in/out laps if not already handled)
    valid_laps = laps_q_df.dropna(subset=["LapTimeSeconds"])
    if valid_laps.empty:
        print("engineer_features: No valid laps with LapTimeSeconds found.")
        # Return a DataFrame with drivers but NaN/default features
        drivers = laps_q_df["Driver"].unique()
        empty_feats_df = pd.DataFrame({"Driver": drivers})
        feature_cols = [
            "q_avg_lap_time",
            "q_std_lap_time",
            "q_best_lap_time",
            "q_lap_count",
            "q_pit_stop_count",
        ]
        for col in feature_cols:
            empty_feats_df[col] = np.nan  # Or 0.0
        return empty_feats_df

    features_list = []
    for driver_abbr in valid_laps["Driver"].unique():
        driver_laps = valid_laps[valid_laps["Driver"] == driver_abbr]

        avg_lap_time = driver_laps["LapTimeSeconds"].mean()
        std_lap_time = driver_laps["LapTimeSeconds"].std()
        best_lap_time = driver_laps["LapTimeSeconds"].min()
        lap_count = len(driver_laps)

        # Count pit stops during qualifying (usually 0 or 1, but good to have)
        # A pit stop is often marked by a 'PitOutTime'
        pit_stop_count = driver_laps["PitOutTime"].notna().sum()

        features_list.append(
            {
                "Driver": driver_abbr,
                "q_avg_lap_time": avg_lap_time,
                "q_std_lap_time": (
                    std_lap_time if pd.notna(std_lap_time) else 0.0
                ),  # Handle cases with 1 lap
                "q_best_lap_time": best_lap_time,
                "q_lap_count": lap_count,
                "q_pit_stop_count": pit_stop_count,
            }
        )

    if not features_list:
        print(
            "engineer_features: No features generated (e.g. no unique drivers with valid laps)."
        )
        return pd.DataFrame()

    final_features_df = pd.DataFrame(features_list)

    # Fill any remaining NaNs if a driver had only 1 lap (std would be NaN) or other edge cases
    final_features_df.fillna(
        0, inplace=True
    )  # Or a more sophisticated imputation if needed

    print(
        f"engineer_features: Generated post-Q features for {len(final_features_df)} drivers."
    )
    return final_features_df


def engineer_features_pre_event(
    year_of_prediction: int, gp_identifier_of_prediction: any, drivers_list: list
) -> pd.DataFrame:
    """
    Engineers features available BEFORE any on-track session of the gp_identifier_of_prediction.
    Uses historical data up to, but not including, the event being predicted.

    Args:
        year_of_prediction: The year of the GP we are making predictions for.
        gp_identifier_of_prediction: The round number or event name of the GP we are predicting for.
        drivers_list: A list of driver abbreviations (e.g., ['VER', 'HAM']) for whom to generate features.

    Returns:
        Pandas DataFrame with 'Driver' column and other pre-event numeric features.
    """
    if not drivers_list:
        print("engineer_features_pre_event: Received empty drivers_list.")
        return pd.DataFrame()

    print(
        f"engineer_features_pre_event: Engineering features for {len(drivers_list)} drivers for {year_of_prediction} GP {gp_identifier_of_prediction}"
    )

    all_driver_event_features = []
    ergast = Ergast()  # Initialize Ergast client for standings

    # Get the event date of the GP we are predicting for, to filter historical data
    target_event_date = None
    try:
        schedule = fastf1.get_event_schedule(year_of_prediction)
        if isinstance(gp_identifier_of_prediction, int):  # Round number
            event_details = schedule[
                schedule["RoundNumber"] == gp_identifier_of_prediction
            ]
        else:  # Event name (try exact match first, then contains)
            event_details = schedule[
                schedule["EventName"] == gp_identifier_of_prediction
            ]
            if event_details.empty:
                event_details = schedule[
                    schedule["EventName"].str.contains(
                        gp_identifier_of_prediction, case=False, na=False
                    )
                ]

        if not event_details.empty:
            target_event_date = pd.to_datetime(event_details["EventDate"].iloc[0])
            print(
                f"Target event date for pre-event features: {target_event_date.date()}"
            )
        else:
            print(
                f"Could not find event details for {gp_identifier_of_prediction} in year {year_of_prediction} schedule."
            )
            raise ValueError("Target event not found in schedule.")

    except Exception as e:
        print(
            f"Could not get target event date for pre-event features: {e}. Cannot generate historical context."
        )
        # Return a DataFrame with default/NaN values for all drivers
        default_features = pd.DataFrame({"Driver": drivers_list})
        # Define expected feature columns here, even if they are all NaN/0
        expected_cols = [
            "hist_avg_finish_last_5",
            "hist_points_total_season",
            "hist_poles_season",
            "hist_wins_season",
            "driver_age",
            "driver_experience_years",
        ]
        for col in expected_cols:
            default_features[col] = np.nan
        return default_features

    for driver_abbr in drivers_list:
        current_driver_features = {"Driver": driver_abbr}

        # --- Initialize features with defaults ---
        current_driver_features["hist_avg_finish_last_5"] = (
            20.0  # Default (worse than last)
        )
        current_driver_features["hist_points_total_season"] = 0.0
        current_driver_features["hist_poles_season"] = 0
        current_driver_features["hist_wins_season"] = 0
        current_driver_features["driver_age"] = (
            np.nan
        )  # Requires fetching driver birth date
        current_driver_features["driver_experience_years"] = 0  # Years in F1

        # --- Feature 1 & 2: Performance in recent races (last 5) & Season Points ---
        race_results_history = []  # List of {'position': X, 'points': Y} dicts

        # Scan current season and optionally previous season for recent form
        # Limit lookback to avoid excessive API calls or slow processing
        for year_to_scan in range(
            year_of_prediction, year_of_prediction - 2, -1
        ):  # Current and previous year
            if year_to_scan < 2000:
                continue  # Ergast/FastF1 data limits

            try:
                past_schedule = fastf1.get_event_schedule(year_to_scan)
                # Iterate events in chronological order to build history correctly
                for _, past_event in past_schedule.sort_values(
                    by="EventDate"
                ).iterrows():
                    past_event_date = pd.to_datetime(past_event["EventDate"])

                    # Only consider events strictly before the target event's date
                    if past_event_date < target_event_date:
                        if past_event["EventFormat"] == "testing":  # Skip testing
                            continue
                        try:
                            # Using Ergast for results can be faster for bulk historical, but FastF1 is also fine
                            # For simplicity, let's use FastF1 session results
                            sess = fastf1.get_session(
                                year_to_scan, past_event["RoundNumber"], "R"
                            )
                            sess.load(
                                results=True,
                                laps=False,
                                telemetry=False,
                                weather=False,
                                messages=False,
                            )  # Only need results

                            if sess.results is not None:
                                driver_result_row = sess.results[
                                    sess.results["Abbreviation"] == driver_abbr
                                ]
                                if not driver_result_row.empty:
                                    pos = pd.to_numeric(
                                        driver_result_row["Position"].iloc[0],
                                        errors="coerce",
                                    )
                                    pts = pd.to_numeric(
                                        driver_result_row["Points"].iloc[0],
                                        errors="coerce",
                                    )

                                    # Store if position is valid, for recency calculations
                                    if pd.notna(pos):
                                        race_results_history.append(
                                            {
                                                "position": pos,
                                                "points": pts if pd.notna(pts) else 0.0,
                                                "year": year_to_scan,
                                            }
                                        )
                        except Exception as e_sess:
                            # print(f"Minor error loading session {year_to_scan} R{past_event['RoundNumber']} for {driver_abbr}: {e_sess}")
                            pass  # Continue if one session fails
                    else:
                        # We've reached or passed the target event date for this year, no need to look further in this year
                        if year_to_scan == year_of_prediction:
                            break
            except Exception as e_sched:
                print(f"Could not load schedule for year {year_to_scan}: {e_sched}")

            if (
                len(race_results_history) >= 5 and year_to_scan == year_of_prediction
            ):  # If we have enough recent races from current year
                break

        # Calculate from collected history (most recent first)
        if race_results_history:
            race_results_history.sort(
                key=lambda x: (x["year"], x.get("round", 0)), reverse=True
            )  # Ensure correct recency

            # Avg finish in last 5 races
            last_5_positions = [r["position"] for r in race_results_history[:5]]
            if last_5_positions:
                current_driver_features["hist_avg_finish_last_5"] = np.mean(
                    last_5_positions
                )

            # Points, poles, wins THIS season
            current_season_results = [
                r for r in race_results_history if r["year"] == year_of_prediction
            ]
            if current_season_results:
                current_driver_features["hist_points_total_season"] = sum(
                    r["points"] for r in current_season_results
                )
                current_driver_features["hist_poles_season"] = sum(
                    1
                    for r in current_season_results
                    if r["position"] == 1 and r.get("grid_position") == 1
                )  # Simplified pole
                current_driver_features["hist_wins_season"] = sum(
                    1 for r in current_season_results if r["position"] == 1
                )

        # --- Feature 3: Driver Age & Experience (Example using Ergast for driver info) ---
        # This is more complex as it requires mapping driver_abbr to Ergast driverId and fetching info.
        # FastF1 session.get_driver() can also provide some info if a recent session is loaded.
        # For a truly pre-event scenario without loading current sessions, Ergast is better.
        try:
            # This part is a simplification. A robust solution needs good driver ID mapping.
            # drivers_info_ergast = ergast.get_driver_info(season=year_of_prediction, driver=driver_abbr.lower()) # Ergast uses lowercase e.g. 'verstappen' not 'VER'
            # For this example, let's assume we have a way to get birthDate.
            # Placeholder, actual implementation would fetch this.
            # driver_birth_year = 1990 # EXAMPLE
            # current_driver_features['driver_age'] = year_of_prediction - driver_birth_year
            # current_driver_features['driver_experience_years'] = year_of_prediction - 2015 # EXAMPLE first year
            pass  # Skipping full implementation for brevity, but this is an area for expansion.
        except Exception as e_ergast:
            # print(f"Could not fetch driver info from Ergast for {driver_abbr}: {e_ergast}")
            pass

        all_driver_event_features.append(current_driver_features)

    if not all_driver_event_features:
        print("engineer_features_pre_event: No features generated for any driver.")
        return pd.DataFrame()

    final_df = pd.DataFrame(all_driver_event_features)

    # Define ALL expected pre-event columns that the model was trained on
    # This ensures consistency even if some features couldn't be calculated for some drivers
    expected_cols_final = [
        "hist_avg_finish_last_5",
        "hist_points_total_season",
        "hist_poles_season",
        "hist_wins_season",
        "driver_age",
        "driver_experience_years",
    ]  # Add ALL your pre-event feature names here

    for col in expected_cols_final:
        if col not in final_df.columns:
            final_df[col] = (
                np.nan
            )  # Add as NaN, preprocessing in predict script will impute

    # Select only the expected columns in the right order if your training script relies on it
    # final_df = final_df[['Driver'] + expected_cols_final] # Optional, but good practice

    print(
        f"engineer_features_pre_event: Generated pre-event features for {len(final_df)} drivers."
    )
    return final_df
