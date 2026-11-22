"""
Circuit profiles, characteristics, and authentic real-world GPS track coordinates.
Loads genuine telemetry GPS coordinates harvested directly from OpenF1 / FastF1 for all 24 F1 circuits.
"""
import os
import json
import re
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TRACK_COORDS_FILE = os.path.join(DATA_DIR, "real_track_coordinates.json")

# Complete roster of all 24 Grand Prix circuits on the modern F1 Calendar
CIRCUITS = {
    "Monza (Italian GP)": {
        "id": "monza",
        "key": "Monza",
        "country": "Italy",
        "length_km": 5.793,
        "laps": 53,
        "type": "Power Circuit / Low Downforce",
        "overtaking_difficulty": "Low (Long Straights & DRS)",
        "tire_degradation": "Medium",
        "safety_car_prob": 0.45,
        "downforce_index": 0.20,
        "corners": 11,
        "drs_zones": 2,
        "avg_speed_kmh": 260.5
    },
    "Silverstone (British GP)": {
        "id": "silverstone",
        "key": "Silverstone",
        "country": "United Kingdom",
        "length_km": 5.891,
        "laps": 52,
        "type": "High Speed / High Lateral Load",
        "overtaking_difficulty": "Medium",
        "tire_degradation": "High (Maggotts-Becketts Complex)",
        "safety_car_prob": 0.55,
        "downforce_index": 0.75,
        "corners": 18,
        "drs_zones": 2,
        "avg_speed_kmh": 245.0
    },
    "Spa-Francorchamps (Belgian GP)": {
        "id": "spa",
        "key": "Spa-Francorchamps",
        "country": "Belgium",
        "length_km": 7.004,
        "laps": 44,
        "type": "Elevation / Mixed High Speed & Sector 2 Downforce",
        "overtaking_difficulty": "Low (Kemmel Straight)",
        "tire_degradation": "High",
        "safety_car_prob": 0.65,
        "downforce_index": 0.60,
        "corners": 19,
        "drs_zones": 2,
        "avg_speed_kmh": 240.2
    },
    "Monaco (Monaco GP)": {
        "id": "monaco",
        "key": "Monte Carlo",
        "country": "Monaco",
        "length_km": 3.337,
        "laps": 78,
        "type": "Street Circuit / Maximum Downforce",
        "overtaking_difficulty": "Extremely High (Qualifying Critical)",
        "tire_degradation": "Low",
        "safety_car_prob": 0.80,
        "downforce_index": 0.98,
        "corners": 19,
        "drs_zones": 1,
        "avg_speed_kmh": 160.8
    },
    "Marina Bay (Singapore GP)": {
        "id": "singapore",
        "key": "Singapore",
        "country": "Singapore",
        "length_km": 4.940,
        "laps": 62,
        "type": "Night Street Circuit / High Downforce & Humidity",
        "overtaking_difficulty": "High",
        "tire_degradation": "High",
        "safety_car_prob": 0.90,
        "downforce_index": 0.90,
        "corners": 19,
        "drs_zones": 3,
        "avg_speed_kmh": 185.0
    },
    "Bahrain (Sakhir GP)": {
        "id": "bahrain",
        "key": "Sakhir",
        "country": "Bahrain",
        "length_km": 5.412,
        "laps": 57,
        "type": "Traction & Heavy Braking",
        "overtaking_difficulty": "Low (Turn 1, 4, 11)",
        "tire_degradation": "Very High (Abrasive Asphalt)",
        "safety_car_prob": 0.50,
        "downforce_index": 0.65,
        "corners": 15,
        "drs_zones": 3,
        "avg_speed_kmh": 215.4
    },
    "Suzuka (Japanese GP)": {
        "id": "suzuka",
        "key": "Suzuka",
        "country": "Japan",
        "length_km": 5.807,
        "laps": 53,
        "type": "Figure-8 / High Aerodynamic Efficiency",
        "overtaking_difficulty": "High (Esses & 130R)",
        "tire_degradation": "High",
        "safety_car_prob": 0.50,
        "downforce_index": 0.85,
        "corners": 18,
        "drs_zones": 1,
        "avg_speed_kmh": 235.0
    },
    "Melbourne (Australian GP)": {
        "id": "melbourne",
        "key": "Melbourne",
        "country": "Australia",
        "length_km": 5.278,
        "laps": 58,
        "type": "Semi-Street / High Speed Flowing",
        "overtaking_difficulty": "Medium",
        "tire_degradation": "Medium",
        "safety_car_prob": 0.70,
        "downforce_index": 0.60,
        "corners": 14,
        "drs_zones": 4,
        "avg_speed_kmh": 238.0
    },
    "Jeddah Corniche (Saudi Arabian GP)": {
        "id": "jeddah",
        "key": "Jeddah",
        "country": "Saudi Arabia",
        "length_km": 6.174,
        "laps": 50,
        "type": "Fastest Street Circuit / Low Downforce",
        "overtaking_difficulty": "Medium",
        "tire_degradation": "Low",
        "safety_car_prob": 0.85,
        "downforce_index": 0.40,
        "corners": 27,
        "drs_zones": 3,
        "avg_speed_kmh": 252.0
    },
    "Miami (Miami GP)": {
        "id": "miami",
        "key": "Miami",
        "country": "United States",
        "length_km": 5.412,
        "laps": 57,
        "type": "Semi-Permanent / Long Straights & Tight Chicane",
        "overtaking_difficulty": "Medium",
        "tire_degradation": "Medium",
        "safety_car_prob": 0.60,
        "downforce_index": 0.55,
        "corners": 19,
        "drs_zones": 3,
        "avg_speed_kmh": 222.0
    },
    "Imola (Emilia Romagna GP)": {
        "id": "imola",
        "key": "Imola",
        "country": "Italy",
        "length_km": 4.909,
        "laps": 63,
        "type": "Classic Old-School / Narrow & Technical",
        "overtaking_difficulty": "High (Main Straight Only)",
        "tire_degradation": "Medium",
        "safety_car_prob": 0.65,
        "downforce_index": 0.75,
        "corners": 19,
        "drs_zones": 1,
        "avg_speed_kmh": 230.0
    },
    "Catalunya (Spanish GP)": {
        "id": "catalunya",
        "key": "Catalunya",
        "country": "Spain",
        "length_km": 4.657,
        "laps": 66,
        "type": "Aerodynamic Benchmark / Fast Final Turn",
        "overtaking_difficulty": "Medium",
        "tire_degradation": "Very High",
        "safety_car_prob": 0.35,
        "downforce_index": 0.80,
        "corners": 14,
        "drs_zones": 2,
        "avg_speed_kmh": 220.0
    },
    "Montreal (Canadian GP)": {
        "id": "montreal",
        "key": "Montreal",
        "country": "Canada",
        "length_km": 4.361,
        "laps": 70,
        "type": "Stop-and-Go / Wall of Champions",
        "overtaking_difficulty": "Low (Hairpin & Final Chicane)",
        "tire_degradation": "Low",
        "safety_car_prob": 0.75,
        "downforce_index": 0.45,
        "corners": 14,
        "drs_zones": 3,
        "avg_speed_kmh": 218.0
    },
    "Red Bull Ring (Austrian GP)": {
        "id": "austria",
        "key": "Spielberg",
        "country": "Austria",
        "length_km": 4.318,
        "laps": 71,
        "type": "Short Elevation / Heavy Braking Zones",
        "overtaking_difficulty": "Low (Turn 1, 3, 4 DRS)",
        "tire_degradation": "Medium",
        "safety_car_prob": 0.50,
        "downforce_index": 0.50,
        "corners": 10,
        "drs_zones": 3,
        "avg_speed_kmh": 242.0
    },
    "Hungaroring (Hungarian GP)": {
        "id": "hungary",
        "key": "Hungaroring",
        "country": "Hungary",
        "length_km": 4.381,
        "laps": 70,
        "type": "Monaco Without Walls / High Downforce",
        "overtaking_difficulty": "High",
        "tire_degradation": "Medium",
        "safety_car_prob": 0.40,
        "downforce_index": 0.90,
        "corners": 14,
        "drs_zones": 2,
        "avg_speed_kmh": 195.0
    },
    "Zandvoort (Dutch GP)": {
        "id": "zandvoort",
        "key": "Zandvoort",
        "country": "Netherlands",
        "length_km": 4.259,
        "laps": 72,
        "type": "Banked Corners / High Downforce Dunes",
        "overtaking_difficulty": "High",
        "tire_degradation": "High",
        "safety_car_prob": 0.60,
        "downforce_index": 0.88,
        "corners": 14,
        "drs_zones": 2,
        "avg_speed_kmh": 210.0
    },
    "Baku City (Azerbaijan GP)": {
        "id": "baku",
        "key": "Baku",
        "country": "Azerbaijan",
        "length_km": 6.003,
        "laps": 51,
        "type": "Longest Straight (2.2km) & Castle Section",
        "overtaking_difficulty": "Low (Slipstream Monster)",
        "tire_degradation": "Low",
        "safety_car_prob": 0.80,
        "downforce_index": 0.45,
        "corners": 20,
        "drs_zones": 2,
        "avg_speed_kmh": 215.0
    },
    "Austin COTA (United States GP)": {
        "id": "cota",
        "key": "Austin",
        "country": "United States",
        "length_km": 5.513,
        "laps": 56,
        "type": "Sector 1 Esses & Turn 1 Crest",
        "overtaking_difficulty": "Medium",
        "tire_degradation": "High",
        "safety_car_prob": 0.50,
        "downforce_index": 0.75,
        "corners": 20,
        "drs_zones": 2,
        "avg_speed_kmh": 220.0
    },
    "Autodromo Hermanos Rodriguez (Mexico GP)": {
        "id": "mexico",
        "key": "Mexico City",
        "country": "Mexico",
        "length_km": 4.304,
        "laps": 71,
        "type": "High Altitude (2200m) / Low Air Density",
        "overtaking_difficulty": "Low (Turn 1 Straight)",
        "tire_degradation": "Medium",
        "safety_car_prob": 0.55,
        "downforce_index": 0.95,
        "corners": 17,
        "drs_zones": 3,
        "avg_speed_kmh": 210.0
    },
    "Interlagos (Sao Paulo GP)": {
        "id": "interlagos",
        "key": "Interlagos",
        "country": "Brazil",
        "length_km": 4.309,
        "laps": 71,
        "type": "Anti-Clockwise Classic / Senna 'S'",
        "overtaking_difficulty": "Low (Turn 1 & 4)",
        "tire_degradation": "High",
        "safety_car_prob": 0.75,
        "downforce_index": 0.65,
        "corners": 15,
        "drs_zones": 2,
        "avg_speed_kmh": 218.0
    },
    "Las Vegas Strip Circuit": {
        "id": "vegas",
        "key": "Las Vegas",
        "country": "United States",
        "length_km": 6.201,
        "laps": 50,
        "type": "Cold Night Street / Massive Strip Straight",
        "overtaking_difficulty": "Low",
        "tire_degradation": "Low (Grain Volatility)",
        "safety_car_prob": 0.70,
        "downforce_index": 0.30,
        "corners": 17,
        "drs_zones": 2,
        "avg_speed_kmh": 240.0
    },
    "Lusail (Qatar GP)": {
        "id": "lusail",
        "key": "Lusail",
        "country": "Qatar",
        "length_km": 5.419,
        "laps": 57,
        "type": "High Speed Sweepers / Extreme Tire Load",
        "overtaking_difficulty": "Medium",
        "tire_degradation": "Extremely High (Pirelli Limit)",
        "safety_car_prob": 0.60,
        "downforce_index": 0.80,
        "corners": 16,
        "drs_zones": 1,
        "avg_speed_kmh": 235.0
    },
    "Yas Marina (Abu Dhabi GP)": {
        "id": "abudhabi",
        "key": "Yas Marina Circuit",
        "country": "United Arab Emirates",
        "length_km": 5.281,
        "laps": 58,
        "type": "Twilight Season Finale / Hotel Sector",
        "overtaking_difficulty": "Medium",
        "tire_degradation": "Medium",
        "safety_car_prob": 0.50,
        "downforce_index": 0.65,
        "corners": 16,
        "drs_zones": 2,
        "avg_speed_kmh": 225.0
    },
    "Shanghai (Chinese GP)": {
        "id": "shanghai",
        "key": "Shanghai",
        "country": "China",
        "length_km": 5.451,
        "laps": 56,
        "type": "Snail Turn 1-2 & 1.2km Back Straight",
        "overtaking_difficulty": "Low (Hairpin Turn 14)",
        "tire_degradation": "High",
        "safety_car_prob": 0.55,
        "downforce_index": 0.70,
        "corners": 16,
        "drs_zones": 2,
        "avg_speed_kmh": 215.0
    },
    "Sepang (Malaysian GP)": {
        "id": "sepang",
        "key": "Sepang",
        "country": "Malaysia",
        "length_km": 5.543,
        "laps": 56,
        "type": "High Aerodynamic Load & Tropical Heat",
        "overtaking_difficulty": "Medium (Turn 1 & Turn 15 Hairpin)",
        "tire_degradation": "Very High (Coarse Tarmac)",
        "safety_car_prob": 0.50,
        "downforce_index": 0.70,
        "corners": 15,
        "drs_zones": 2,
        "avg_speed_kmh": 210.5
    }
}

RACE_TO_CIRCUIT_MAP = {
    "bahrain grand prix in malaysia": "Sepang (Malaysian GP)",
    "malaysia": "Sepang (Malaysian GP)",
    "sepang": "Sepang (Malaysian GP)",
    "bahrain": "Bahrain (Sakhir GP)",
    "saudi": "Jeddah Corniche (Saudi Arabian GP)",
    "jeddah": "Jeddah Corniche (Saudi Arabian GP)",
    "australian": "Melbourne (Australian GP)",
    "melbourne": "Melbourne (Australian GP)",
    "emilia": "Imola (Emilia Romagna GP)",
    "imola": "Imola (Emilia Romagna GP)",
    "miami": "Miami (Miami GP)",
    "spanish": "Catalunya (Spanish GP)",
    "barcelona": "Catalunya (Spanish GP)",
    "catalunya": "Catalunya (Spanish GP)",
    "monaco": "Monaco (Monaco GP)",
    "monte carlo": "Monaco (Monaco GP)",
    "azerbaijan": "Baku City (Azerbaijan GP)",
    "baku": "Baku City (Azerbaijan GP)",
    "canadian": "Montreal (Canadian GP)",
    "montreal": "Montreal (Canadian GP)",
    "british": "Silverstone (British GP)",
    "silverstone": "Silverstone (British GP)",
    "austrian": "Red Bull Ring (Austrian GP)",
    "spielberg": "Red Bull Ring (Austrian GP)",
    "hungarian": "Hungaroring (Hungarian GP)",
    "hungaroring": "Hungaroring (Hungarian GP)",
    "belgian": "Spa-Francorchamps (Belgian GP)",
    "spa": "Spa-Francorchamps (Belgian GP)",
    "dutch": "Zandvoort (Dutch GP)",
    "zandvoort": "Zandvoort (Dutch GP)",
    "italian": "Monza (Italian GP)",
    "monza": "Monza (Italian GP)",
    "singapore": "Marina Bay (Singapore GP)",
    "japanese": "Suzuka (Japanese GP)",
    "suzuka": "Suzuka (Japanese GP)",
    "qatar": "Lusail (Qatar GP)",
    "lusail": "Lusail (Qatar GP)",
    "losail": "Lusail (Qatar GP)",
    "united states": "Austin COTA (United States GP)",
    "cota": "Austin COTA (United States GP)",
    "austin": "Austin COTA (United States GP)",
    "mexico": "Autodromo Hermanos Rodriguez (Mexico GP)",
    "são paulo": "Interlagos (Sao Paulo GP)",
    "sao paulo": "Interlagos (Sao Paulo GP)",
    "brazilian": "Interlagos (Sao Paulo GP)",
    "interlagos": "Interlagos (Sao Paulo GP)",
    "las vegas": "Las Vegas Strip Circuit",
    "abu dhabi": "Yas Marina (Abu Dhabi GP)",
    "yas marina": "Yas Marina (Abu Dhabi GP)",
    "chinese": "Shanghai (Chinese GP)",
    "shanghai": "Shanghai (Chinese GP)"
}

def get_circuit_by_race_name(race_name: str):
    """Matches any official Grand Prix race name to its circuit definition."""
    name_lower = race_name.lower()
    for key, c_name in sorted(RACE_TO_CIRCUIT_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if key in name_lower:
            return c_name, CIRCUITS[c_name]
    # If not recognized, return None
    return None, None

_CACHED_TRACK_DB = None
_CACHED_TRACK_MTIME = 0

def load_track_db():
    global _CACHED_TRACK_DB, _CACHED_TRACK_MTIME
    if os.path.exists(TRACK_COORDS_FILE):
        try:
            mtime = os.path.getmtime(TRACK_COORDS_FILE)
            if _CACHED_TRACK_DB is not None and mtime == _CACHED_TRACK_MTIME:
                return _CACHED_TRACK_DB
            with open(TRACK_COORDS_FILE, "r") as f:
                _CACHED_TRACK_DB = json.load(f)
                _CACHED_TRACK_MTIME = mtime
                return _CACHED_TRACK_DB
        except Exception:
            pass
    if _CACHED_TRACK_DB is not None:
        return _CACHED_TRACK_DB
    _CACHED_TRACK_DB = {}
    return _CACHED_TRACK_DB

def generate_track_geometry(circuit_id_or_key: str, n_points: int = 300):
    db = load_track_db()
    
    matched_data = None
    # 1. Exact match in db keys
    if circuit_id_or_key in db:
        matched_data = db[circuit_id_or_key]
        
    # 2. Match via CIRCUITS catalog
    if matched_data is None:
        for c_name, c_info in CIRCUITS.items():
            if c_info["id"] == circuit_id_or_key or c_info["key"] == circuit_id_or_key or c_name == circuit_id_or_key:
                matched_data = db.get(c_info["key"]) or db.get(c_info.get("id")) or db.get(c_name)
                break

    # 3. Fuzzy search across db keys
    if matched_data is None:
        target_lower = circuit_id_or_key.lower()
        for k, v in db.items():
            if k.lower() in target_lower or target_lower in k.lower():
                matched_data = v
                break

    if matched_data and "x" in matched_data and "y" in matched_data:
        raw_x = np.array(matched_data["x"], dtype=float)
        raw_y = np.array(matched_data["y"], dtype=float)

        if np.hypot(raw_x[0] - raw_x[-1], raw_y[0] - raw_y[-1]) > 50:
            raw_x = np.append(raw_x, raw_x[0])
            raw_y = np.append(raw_y, raw_y[0])

        x_min, x_max = np.min(raw_x), np.max(raw_x)
        y_min, y_max = np.min(raw_y), np.max(raw_y)
        x_range = max(1.0, x_max - x_min)
        y_range = max(1.0, y_max - y_min)
        scale = max(x_range, y_range)

        norm_x = (raw_x - x_min) / scale * 1000.0
        norm_y = (raw_y - y_min) / scale * 1000.0
        return norm_x, norm_y

    t = np.linspace(0, 2 * np.pi, n_points)
    x = 500 + 400 * np.cos(t)
    y = 500 + 300 * np.sin(t)
    return x, y
