import pandas as pd


def engineer_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Creates aggregate race-level features for ML.
    """
    # average lap time
    avg_lap = laps.groupby("Driver")["LapTime"].mean().rename("avg_lap_time")
    # lap time consistency (std dev)
    std_lap = laps.groupby("Driver")["LapTime"].std().rename("std_lap_time")
    # number of pit stops
    pit_stops = (
        laps[laps["PitOutTime"].notnull()].groupby("Driver").size().rename("pit_count")
    )
    # merge into one DataFrame
    features = pd.concat([avg_lap, std_lap, pit_stops], axis=1).fillna(0)
    return features.reset_index()
