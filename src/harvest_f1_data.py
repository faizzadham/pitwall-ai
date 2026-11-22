"""
F1 Real-World Data Harvester (High Performance Batch Mode).
1. Authentic real-world GPS track coordinates for all 24 F1 circuits from OpenF1.
2. Genuine Grand Prix race results from 2022 to recent seasons (2022-2026) via Ergast/Jolpi batching.
3. Feature engineering: rolling driver form, team momentum, circuit downforce, and tire degradation.
"""
import os
import json
import time
import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
TRACK_COORDS_FILE = os.path.join(DATA_DIR, "real_track_coordinates.json")
RESULTS_CSV_FILE = os.path.join(DATA_DIR, "f1_historical_processed.csv")

# OpenF1 Session Keys for authentic GPS telemetry extraction
CIRCUIT_SESSION_MAP = {
    "Sakhir": 7767,
    "Jeddah": 7774,
    "Melbourne": 7782,
    "Baku": 9064,
    "Miami": 9073,
    "Imola": 9515,
    "Monte Carlo": 9089,
    "Catalunya": 9097,
    "Montreal": 9105,
    "Spielberg": 9112,
    "Silverstone": 9121,
    "Hungaroring": 9128,
    "Spa-Francorchamps": 9135,
    "Zandvoort": 9144,
    "Monza": 9152,
    "Singapore": 9160,
    "Suzuka": 9168,
    "Lusail": 9215,
    "Austin": 9207,
    "Mexico City": 9176,
    "Interlagos": 9304,
    "Las Vegas": 9184,
    "Yas Marina Circuit": 9192,
    "Shanghai": 9664
}

def harvest_real_track_coordinates():
    """Fetches authentic GPS track coordinates for all F1 circuits from OpenF1."""
    print("=== HARVESTING AUTHENTIC REAL TRACK COORDINATES FROM OPENF1 ===", flush=True)
    track_db = {}
    
    if os.path.exists(TRACK_COORDS_FILE):
        try:
            with open(TRACK_COORDS_FILE, "r") as f:
                track_db = json.load(f)
        except Exception:
            track_db = {}

    for circuit_name, s_key in CIRCUIT_SESSION_MAP.items():
        if circuit_name in track_db and len(track_db[circuit_name].get("x", [])) > 50:
            print(f"  [Cached] {circuit_name}: {len(track_db[circuit_name]['x'])} points", flush=True)
            continue

        print(f"  Fetching {circuit_name} (Session {s_key})...", flush=True)
        coords_found = False

        for d_num in [1, 16, 4, 44, 55, 63, 11, 81]:
            try:
                laps_url = f"https://api.openf1.org/v1/laps?session_key={s_key}&driver_number={d_num}"
                r_laps = requests.get(laps_url, timeout=6).json()
                df_laps = pd.DataFrame(r_laps).dropna(subset=["lap_duration"])
                
                if df_laps.empty:
                    continue

                fastest = df_laps.sort_values("lap_duration").iloc[0]
                s_dt = pd.to_datetime(fastest["date_start"])
                dur = float(fastest["lap_duration"])
                e_dt = s_dt + pd.to_timedelta(dur, unit='s')

                s_str = s_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                e_str = e_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

                loc_url = f"https://api.openf1.org/v1/location?session_key={s_key}&driver_number={d_num}&date>={s_str}&date<={e_str}"
                r_loc = requests.get(loc_url, timeout=8).json()
                df_loc = pd.DataFrame(r_loc)

                if not df_loc.empty and "x" in df_loc.columns and "y" in df_loc.columns:
                    valid = df_loc[(df_loc["x"] != 0) | (df_loc["y"] != 0)]
                    if len(valid) > 40:
                        x_pts = valid["x"].tolist()
                        y_pts = valid["y"].tolist()
                        track_db[circuit_name] = {
                            "circuit": circuit_name,
                            "x": x_pts,
                            "y": y_pts,
                            "points": len(x_pts)
                        }
                        coords_found = True
                        print(f"    -> Success with Driver {d_num}: {len(x_pts)} GPS points!", flush=True)
                        break
            except Exception:
                time.sleep(0.15)
                continue

    with open(TRACK_COORDS_FILE, "w") as f:
        json.dump(track_db, f, indent=2)
    print(f"Saved {len(track_db)} authentic track coordinates to {TRACK_COORDS_FILE}\n", flush=True)
    return track_db

def harvest_real_race_results(seasons=[2022, 2023, 2024, 2025, 2026]):
    """Harvests genuine Grand Prix race results from 2022 to recent seasons using batch pagination."""
    print("=== HARVESTING GENUINE RACE RESULTS (2022-2026) FROM ERGAST/JOLPI ===", flush=True)
    all_rows = []

    circuit_profiles = {
        "Monza": {"downforce": 0.20, "tire_deg": 0.50},
        "Silverstone": {"downforce": 0.75, "tire_deg": 0.80},
        "Spa": {"downforce": 0.60, "tire_deg": 0.75},
        "Monaco": {"downforce": 0.98, "tire_deg": 0.30},
        "Singapore": {"downforce": 0.90, "tire_deg": 0.85},
        "Sakhir": {"downforce": 0.65, "tire_deg": 0.85},
        "Suzuka": {"downforce": 0.85, "tire_deg": 0.80},
        "Zandvoort": {"downforce": 0.88, "tire_deg": 0.70},
        "Hungaroring": {"downforce": 0.90, "tire_deg": 0.65},
        "Red Bull Ring": {"downforce": 0.50, "tire_deg": 0.70},
        "Spielberg": {"downforce": 0.50, "tire_deg": 0.70},
        "Catalunya": {"downforce": 0.80, "tire_deg": 0.75},
        "Albert Park": {"downforce": 0.60, "tire_deg": 0.60},
        "Melbourne": {"downforce": 0.60, "tire_deg": 0.60},
        "Jeddah": {"downforce": 0.40, "tire_deg": 0.50},
        "Miami": {"downforce": 0.55, "tire_deg": 0.65},
        "Baku": {"downforce": 0.45, "tire_deg": 0.50},
        "Montreal": {"downforce": 0.45, "tire_deg": 0.55},
        "Austin": {"downforce": 0.75, "tire_deg": 0.80},
        "Mexico": {"downforce": 0.95, "tire_deg": 0.60},
        "Interlagos": {"downforce": 0.65, "tire_deg": 0.75},
        "Las Vegas": {"downforce": 0.30, "tire_deg": 0.40},
        "Losail": {"downforce": 0.80, "tire_deg": 0.90},
        "Lusail": {"downforce": 0.80, "tire_deg": 0.90},
        "Yas Marina": {"downforce": 0.65, "tire_deg": 0.65},
        "Shanghai": {"downforce": 0.70, "tire_deg": 0.70},
        "Imola": {"downforce": 0.75, "tire_deg": 0.65}
    }

    driver_history = {}
    team_history = {}

    for year in seasons:
        print(f"Fetching Season {year}...", flush=True)
        season_races = []
        
        for offset in range(0, 600, 100):
            try:
                url = f"https://api.jolpi.ca/ergast/f1/{year}/results.json?limit=100&offset={offset}"
                resp = requests.get(url, timeout=8).json()
                races = resp.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if not races:
                    break
                season_races.extend(races)
            except Exception as e:
                print(f"  Warning on {year} offset {offset}: {e}", flush=True)
                break

        print(f"  -> Retrieved {len(season_races)} Grand Prix races for {year}", flush=True)

        for race in season_races:
            rnd = int(race.get("round", 1))
            race_name = race.get("raceName", f"Round {rnd}")
            circuit_name = race.get("Circuit", {}).get("circuitName", "Unknown")
            results = race.get("Results", [])

            profile = {"downforce": 0.65, "tire_deg": 0.65}
            for key, prof in circuit_profiles.items():
                if key.lower() in circuit_name.lower() or key.lower() in race_name.lower():
                    profile = prof
                    break

            for res in results:
                driver = res.get("Driver", {})
                d_code = driver.get("code") or driver.get("driverId", "UNK")[:3].upper()
                team = res.get("Constructor", {}).get("name", "Unknown")
                grid = int(res.get("grid", 10))
                if grid == 0:
                    grid = 20

                pos_text = res.get("positionText", "20")
                pos = int(res.get("position", 20)) if pos_text.isdigit() else 20
                pts = float(res.get("points", 0))

                if d_code not in driver_history:
                    driver_history[d_code] = []
                if team not in team_history:
                    team_history[team] = []

                recent_driver_pos = np.mean(driver_history[d_code][-5:]) if len(driver_history[d_code]) > 0 else float(grid)
                recent_team_pts = np.mean(team_history[team][-3:]) if len(team_history[team]) > 0 else 5.0

                driver_history[d_code].append(pos)
                team_history[team].append(pts)

                all_rows.append({
                    "season": int(year),
                    "round": int(rnd),
                    "circuit": race_name,
                    "driver": d_code,
                    "team": team,
                    "grid_position": grid,
                    "driver_rolling_avg_pos": round(float(recent_driver_pos), 2),
                    "team_rolling_pts": round(float(recent_team_pts), 2),
                    "circuit_downforce": profile["downforce"],
                    "tire_deg_factor": profile["tire_deg"],
                    "rain_prob": 0.15 if "wet" in res.get("status", "").lower() else 0.05,
                    "final_position": pos,
                    "podium_finish": 1 if pos <= 3 else 0,
                    "race_winner": 1 if pos == 1 else 0
                })

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(RESULTS_CSV_FILE, index=False)
        print(f"\nSUCCESS: Harvested {len(df)} genuine F1 race records across seasons {seasons}!", flush=True)
        print(f"Saved to: {RESULTS_CSV_FILE}", flush=True)
        return df
    else:
        print("No rows harvested.", flush=True)
        return None

if __name__ == "__main__":
    harvest_real_track_coordinates()
    harvest_real_race_results()
