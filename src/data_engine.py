"""
Data acquisition and processing engine.
Supports FastF1 caching, realistic multi-channel telemetry generator, and multi-season race datasets.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CACHE_DIR = os.path.join(DATA_DIR, "cache")
HIST_DATA_FILE = os.path.join(DATA_DIR, "f1_historical_processed.csv")
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    import fastf1
    if hasattr(fastf1.Cache, "configure"):
        fastf1.Cache.configure(cache_dir=CACHE_DIR)
    else:
        fastf1.Cache.enable_cache(CACHE_DIR)
    FASTF1_AVAILABLE = True
except Exception:
    FASTF1_AVAILABLE = False

CURRENT_ROSTER = [
    {"driver": "VER", "name": "Max Verstappen", "team": "Red Bull Racing", "base_pace": 9.6},
    {"driver": "NOR", "name": "Lando Norris", "team": "McLaren", "base_pace": 9.5},
    {"driver": "PIA", "name": "Oscar Piastri", "team": "McLaren", "base_pace": 9.3},
    {"driver": "LEC", "name": "Charles Leclerc", "team": "Ferrari", "base_pace": 9.4},
    {"driver": "HAM", "name": "Lewis Hamilton", "team": "Ferrari", "base_pace": 9.3},
    {"driver": "RUS", "name": "George Russell", "team": "Mercedes", "base_pace": 9.2},
    {"driver": "ANT", "name": "Kimi Antonelli", "team": "Mercedes", "base_pace": 8.8},
    {"driver": "ALO", "name": "Fernando Alonso", "team": "Aston Martin", "base_pace": 8.6},
    {"driver": "STR", "name": "Lance Stroll", "team": "Aston Martin", "base_pace": 8.0},
    {"driver": "GAS", "name": "Pierre Gasly", "team": "Alpine", "base_pace": 8.2},
    {"driver": "DOO", "name": "Jack Doohan", "team": "Alpine", "base_pace": 7.9},
    {"driver": "ALB", "name": "Alexander Albon", "team": "Williams", "base_pace": 8.3},
    {"driver": "SAI", "name": "Carlos Sainz", "team": "Williams", "base_pace": 9.0},
    {"driver": "TSU", "name": "Yuki Tsunoda", "team": "Racing Bulls", "base_pace": 8.1},
    {"driver": "LAW", "name": "Liam Lawson", "team": "Racing Bulls", "base_pace": 8.1},
    {"driver": "HUL", "name": "Nico Hulkenberg", "team": "Sauber / Audi", "base_pace": 8.2},
    {"driver": "BOR", "name": "Gabriel Bortoleto", "team": "Sauber / Audi", "base_pace": 7.9},
    {"driver": "OCO", "name": "Esteban Ocon", "team": "Haas", "base_pace": 8.0},
    {"driver": "BEA", "name": "Oliver Bearman", "team": "Haas", "base_pace": 8.1},
    {"driver": "PER", "name": "Sergio Perez", "team": "Red Bull Racing", "base_pace": 8.5},
]

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Red Bull": "#3671C6",
    "McLaren": "#FF8000",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "Racing Bulls": "#6692FF",
    "RB": "#6692FF",
    "Sauber / Audi": "#52E252",
    "Sauber": "#52E252",
    "Haas": "#B6BABD",
    "Haas F1 Team": "#B6BABD"
}

def get_latest_dataset():
    """Loads existing harvested dataset or generates fallback."""
    csv_path = os.path.join(DATA_DIR, "f1_historical_processed.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    # Fallback generator if harvest has not yet finished
    from src.harvest_f1_data import harvest_real_race_results
    return harvest_real_race_results()

def generate_telemetry_trace(driver_1="VER", driver_2="NOR", distance_m=5000, n_points=250):
    """
    Generates high-frequency telemetry speed traces (km/h vs distance)
    comparing two drivers, inspired by f1-race-replay telemetry line graphs.
    """
    dist = np.linspace(0, distance_m, n_points)
    
    # Base track profile: straights, braking zones, apexes
    profile = (
        260
        + 60 * np.sin(dist / 400.0)
        - 80 * (np.sin(dist / 180.0) ** 4)
        + 30 * np.cos(dist / 700.0)
    )
    profile = np.clip(profile, 75, 335)

    noise_1 = np.random.normal(0, 2.0, n_points)
    noise_2 = np.random.normal(0, 2.2, n_points)

    speed_1 = profile + noise_1
    speed_2 = np.roll(profile, 2) + 1.8 * np.cos(dist / 320.0) + noise_2

    throttle_1 = np.clip((speed_1 - 100) / 200 * 100, 0, 100)
    throttle_2 = np.clip((speed_2 - 100) / 200 * 100, 0, 100)
    brake_1 = np.clip((180 - speed_1) / 80 * 100, 0, 100)
    brake_2 = np.clip((180 - speed_2) / 80 * 100, 0, 100)

    gear_1 = np.clip(np.digitize(speed_1, [90, 130, 170, 210, 250, 290, 315]) + 1, 1, 8)
    gear_2 = np.clip(np.digitize(speed_2, [90, 130, 170, 210, 250, 290, 315]) + 1, 1, 8)

    df_telemetry = pd.DataFrame({
        "distance": dist,
        f"{driver_1}_speed": speed_1,
        f"{driver_2}_speed": speed_2,
        f"{driver_1}_throttle": throttle_1,
        f"{driver_2}_throttle": throttle_2,
        f"{driver_1}_brake": brake_1,
        f"{driver_2}_brake": brake_2,
        f"{driver_1}_gear": gear_1,
        f"{driver_2}_gear": gear_2,
    })
    return df_telemetry
