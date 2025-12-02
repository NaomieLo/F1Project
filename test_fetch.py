# test_fetch.py
from f1_data import setup_fastf1, fetch_session_data

# turn on caching
setup_fastf1("cache")

# fetch Monza 2024 race data
df = fetch_session_data(2024, "Monza", "R")


# 1) Inspect columns & types
print(df.columns.tolist())
print(df.info())

# 2) Convert LapTime to a numeric seconds column
df["lap_sec"] = df["LapTime"].dt.total_seconds()

# 3) Back‑fill weather NaNs
df[["TrackTemp", "Rainfall", "WindSpeed", "WindDirection"]] = df[
    ["TrackTemp", "Rainfall", "WindSpeed", "WindDirection"]
].ffill()

# 4) Peek at the cleaned data
print(df[["Driver", "lap_sec", "TrackTemp", "Rainfall"]].head())
