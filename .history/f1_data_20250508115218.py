import fastf1
from fastf1 import Cache
import pandas as pd


def setup_fastf1(cache_dir: str = "cache"):
    """
    Enable caching to speed up repeated requests.
    """
    Cache.enable_cache(cache_dir)


def fetch_session_data(year: int, gp: str, session: str) -> pd.DataFrame:
    """
    Fetch lap and weather data for a given session.
    Args:
        year: e.g., 2024
        gp: Grand Prix name, e.g., 'Monza'
        session: 'Q' (Qualifying), 'R' (Race)
    Returns:
        DataFrame of laps with weather info.
    """
    fastf1.Cache.enable_cache("cache")
    sess = fastf1.get_session(year, gp, session)
    sess.load(telemetry=False, laps=True, weather=True)
    laps = sess.laps
    # merge weather info onto laps
    weather = sess.weather_data
    laps = laps.merge(weather, left_on="Time", right_on="Time", how="left")
    return laps
