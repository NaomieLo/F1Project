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
