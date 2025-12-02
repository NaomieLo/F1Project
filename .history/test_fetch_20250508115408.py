# test_fetch.py
from f1_data import setup_fastf1, fetch_session_data

# turn on caching
setup_fastf1("cache")

# fetch Monza 2024 race data
df = fetch_session_data(2024, "Monza", "R")

# print first 5 rows
print(df.head())
