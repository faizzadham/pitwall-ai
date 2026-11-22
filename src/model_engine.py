"""
Machine Learning Model Engine for PitWall AI.
Trains and deploys models for F1 race outcome prediction:
1. Podium Probability Classifier (Random Forest)
2. Expected Finishing Position Regressor (Gradient Boosting)
3. Explainable Feature Importance extractor
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score
from src.data_engine import get_latest_dataset, CURRENT_ROSTER, DATA_DIR

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODELS_DIR, "f1_predictor.pkl")

FEATURES = [
    "grid_position",
    "driver_rolling_avg_pos",
    "team_rolling_pts",
    "circuit_downforce",
    "tire_deg_factor",
    "rain_prob",
]

class F1Predictor:
    def __init__(self):
        self.podium_model = None
        self.pos_model = None
        self.feature_importance = {}
        self.metrics = {}

    def train(self, df: pd.DataFrame = None):
        if df is None:
            df = get_latest_dataset()

        max_season = df["season"].max()
        train_df = df[df["season"] < max_season].copy()
        test_df = df[df["season"] >= max_season].copy()

        if len(test_df) == 0:
            train_df = df.iloc[:int(len(df) * 0.8)].copy()
            test_df = df.iloc[int(len(df) * 0.8):].copy()

        X_train = train_df[FEATURES]
        y_train_podium = train_df["podium_finish"]
        y_train_pos = train_df["final_position"]

        X_test = test_df[FEATURES]
        y_test_podium = test_df["podium_finish"]
        y_test_pos = test_df["final_position"]

        # 1. Podium Classifier
        self.podium_model = RandomForestClassifier(
            n_estimators=140,
            max_depth=6,
            min_samples_split=4,
            random_state=42
        )
        self.podium_model.fit(X_train, y_train_podium)
        podium_preds = self.podium_model.predict(X_test)

        # 2. Position Regressor
        self.pos_model = GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.07,
            max_depth=4,
            random_state=42
        )
        self.pos_model.fit(X_train, y_train_pos)
        pos_preds = self.pos_model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test_podium, podium_preds)
        prec = precision_score(y_test_podium, podium_preds, zero_division=0)
        mae = mean_absolute_error(y_test_pos, pos_preds)

        self.metrics = {
            "test_accuracy": round(float(acc) * 100, 2),
            "podium_precision": round(float(prec) * 100, 2),
            "position_mae": round(float(mae), 2),
            "train_samples": len(train_df),
            "test_samples": len(test_df)
        }

        # Feature Importance
        importances = self.podium_model.feature_importances_
        self.feature_importance = {
            feat: round(float(imp) * 100, 2)
            for feat, imp in zip(FEATURES, importances)
        }

        self.save()
        return self.metrics

    def save(self):
        payload = {
            "podium_model": self.podium_model,
            "pos_model": self.pos_model,
            "feature_importance": self.feature_importance,
            "metrics": self.metrics,
            "features": FEATURES
        }
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(payload, f)

    def load(self):
        if not os.path.exists(MODEL_FILE):
            return False
        try:
            with open(MODEL_FILE, "rb") as f:
                payload = pickle.load(f)
                self.podium_model = payload["podium_model"]
                self.pos_model = payload["pos_model"]
                self.feature_importance = payload["feature_importance"]
                self.metrics = payload["metrics"]
                return True
        except Exception:
            return False

    def predict_race(self, starting_grid: list, circuit_info: dict, is_rain: bool = False):
        if self.podium_model is None or self.pos_model is None:
            if not self.load():
                self.train()

        input_rows = []
        for d in starting_grid:
            input_rows.append({
                "grid_position": d.get("grid_position", 10),
                "driver_rolling_avg_pos": d.get("driver_rolling_avg_pos", 8.0),
                "team_rolling_pts": d.get("team_rolling_pts", 12.0),
                "circuit_downforce": circuit_info.get("downforce_index", 0.6),
                "tire_deg_factor": 0.8 if circuit_info.get("tire_degradation") == "High" else 0.5,
                "rain_prob": 0.85 if is_rain else 0.05,
            })

        X_pred = pd.DataFrame(input_rows)[FEATURES]

        prob_matrix = self.podium_model.predict_proba(X_pred)
        podium_probs = prob_matrix[:, 1] if prob_matrix.shape[1] > 1 else prob_matrix[:, 0]
        expected_positions = self.pos_model.predict(X_pred)

        results = []
        for i, d in enumerate(starting_grid):
            results.append({
                "driver": d["driver"],
                "name": d.get("name", d["driver"]),
                "team": d["team"],
                "grid_position": d["grid_position"],
                "podium_prob": float(podium_probs[i]),
                "expected_position": float(expected_positions[i])
            })

        out_df = pd.DataFrame(results)
        out_df = out_df.sort_values(by=["expected_position", "podium_prob"], ascending=[True, False]).reset_index(drop=True)
        out_df["predicted_rank"] = range(1, len(out_df) + 1)

        # Normalize Win Probability based on P1 dominance curve
        top_scores = np.exp(-0.75 * (out_df["expected_position"] - 1.0)) * (out_df["podium_prob"] + 0.1)
        out_df["win_prob"] = top_scores / np.sum(top_scores)

        return out_df

    def predict_qualifying(self, starting_grid: list, circuit_info: dict, is_rain: bool = False) -> pd.DataFrame:
        """Predicts qualifying single-lap flying pace, pole position, lap times, and Q1/Q2/Q3 cutoffs."""
        base_lap_sec = (circuit_info.get("length_km", 5.0) / max(100.0, circuit_info.get("avg_speed_kmh", 220.0))) * 3600.0
        if is_rain:
            base_lap_sec += 9.5

        # Sort grid by driver form and team strength with qualifying sensitivity
        sorted_drivers = sorted(
            starting_grid,
            key=lambda d: (
                d.get("grid_position", 10) * 0.45 +
                d.get("driver_rolling_avg_pos", 8.0) * 0.40 -
                (d.get("team_rolling_pts", 10.0) / 10.0) * 0.15
            )
        )

        rows = []
        p1_time = base_lap_sec
        for rank, d in enumerate(sorted_drivers, start=1):
            if rank == 1:
                delta = 0.0
            else:
                step = 0.052 + (rank * 0.018)
                delta = round((rank - 1) * step + (np.sin(rank * 1.5) * 0.025), 3)

            lap_sec = p1_time + delta
            mins = int(lap_sec // 60)
            secs = lap_sec % 60
            time_str = f"{mins}:{secs:06.3f}"
            gap_str = "POLE" if rank == 1 else f"+{delta:.3f}s"

            if rank <= 10:
                phase = "🟢 Q3 Shootout (Top 10)"
            elif rank <= 15:
                phase = "🟡 Q2 Elimination"
            else:
                phase = "🔴 Q1 Elimination"

            rows.append({
                "q_rank": rank,
                "driver": d["driver"],
                "team": d["team"],
                "lap_time": time_str,
                "gap_to_pole": gap_str,
                "delta_sec": delta,
                "phase": phase,
                "grid_position": d.get("grid_position", rank)
            })

        return pd.DataFrame(rows)

    def predict_sprint(self, starting_grid: list, circuit_info: dict, is_rain: bool = False) -> pd.DataFrame:
        """Predicts Sprint race results and points distribution (100km flat-out sprint)."""
        race_pred = self.predict_race(starting_grid, circuit_info, is_rain)
        sprint_df = race_pred.copy()
        sprint_df["sprint_rank"] = range(1, len(sprint_df) + 1)
        
        # Sprint points under official FIA regulations: P1=8, P2=7 ... P8=1, P9+=0
        pts_map = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
        sprint_df["sprint_points"] = sprint_df["sprint_rank"].map(lambda r: pts_map.get(r, 0))
        return sprint_df

    def predict_practice(self, starting_grid: list, circuit_info: dict, fp_session: str = "FP2") -> pd.DataFrame:
        """Simulates Free Practice telemetry benchmarks: Qualifying trim vs Long-run race pace."""
        base_lap_sec = (circuit_info.get("length_km", 5.0) / max(100.0, circuit_info.get("avg_speed_kmh", 220.0))) * 3600.0
        
        rows = []
        for rank, d in enumerate(starting_grid, start=1):
            pos_factor = d.get("driver_rolling_avg_pos", 10.0) / 10.0
            qual_trim_sec = base_lap_sec + (pos_factor * 1.25)
            race_trim_sec = qual_trim_sec + 3.8
            
            s1 = round(qual_trim_sec * 0.31, 3)
            s2 = round(qual_trim_sec * 0.42, 3)
            s3 = round(qual_trim_sec * 0.27, 3)
            
            speed_trap = round(circuit_info.get("avg_speed_kmh", 220.0) * 1.32 - (pos_factor * 4.0), 1)

            rows.append({
                "driver": d["driver"],
                "team": d["team"],
                "flying_lap_sec": round(qual_trim_sec, 3),
                "long_run_pace_sec": round(race_trim_sec, 3),
                "sector_1": s1,
                "sector_2": s2,
                "sector_3": s3,
                "speed_trap_kmh": speed_trap,
                "laps_completed": 18 + (abs(hash(d["driver"])) % 12)
            })
            
        df = pd.DataFrame(rows).sort_values("flying_lap_sec").reset_index(drop=True)
        df["fp_rank"] = range(1, len(df) + 1)
        p1_sec = df.iloc[0]["flying_lap_sec"]
        
        def format_lap(sec):
            m = int(sec // 60)
            s = sec % 60
            return f"{m}:{s:06.3f}"

        df["flying_lap_time"] = df["flying_lap_sec"].apply(format_lap)
        df["long_run_time"] = df["long_run_pace_sec"].apply(format_lap)
        df["gap_to_p1"] = df["flying_lap_sec"].apply(lambda t: "P1" if t == p1_sec else f"+{t - p1_sec:.3f}s")
        return df

def get_default_starting_grid():
    grid = []
    for idx, d in enumerate(CURRENT_ROSTER):
        grid.append({
            "driver": d["driver"],
            "name": d["name"],
            "team": d["team"],
            "grid_position": idx + 1,
            "driver_rolling_avg_pos": max(1.0, 10.0 - (d["base_pace"] - 7.5) * 4.0),
            "team_rolling_pts": max(0.0, (d["base_pace"] - 7.5) * 15.0),
        })
    return grid
