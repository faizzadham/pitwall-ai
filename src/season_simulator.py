"""
Season Simulation Engine for PitWall AI.
Simulates the remaining 2026 Formula 1 season round-by-round until the Abu Dhabi season finale,
predicting race winners, podium finishes, and cumulative World Championship standings (WDC & WCC).
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.circuits import CIRCUITS
from src.data_engine import CURRENT_ROSTER, TEAM_COLORS
from src.model_engine import F1Predictor

# Official Updated 2026 Formula 1 Calendar (24 Championship Rounds)
SEASON_2026_CALENDAR = [
    {
        "round": 1,
        "race_name": "Australian Grand Prix",
        "circuit_label": "Melbourne (Australian GP)",
        "country": "Australia",
        "flag": "🇦🇺",
        "date": "March 15, 2026",
        "type": "Semi-Street / Rapid Chicanes (Season Opener)"
    },
    {
        "round": 2,
        "race_name": "Chinese Grand Prix",
        "circuit_label": "Shanghai (Chinese GP)",
        "country": "China",
        "flag": "🇨🇳",
        "date": "March 22, 2026",
        "type": "Front-Limited / 1.2km Back Straight"
    },
    {
        "round": 3,
        "race_name": "Japanese Grand Prix",
        "circuit_label": "Suzuka (Japanese GP)",
        "country": "Japan",
        "flag": "🇯🇵",
        "date": "April 5, 2026",
        "type": "Figure-8 / High Aerodynamic Load"
    },
    {
        "round": 4,
        "race_name": "Miami Grand Prix",
        "circuit_label": "Miami (Miami GP)",
        "country": "United States",
        "flag": "🇺🇸",
        "date": "May 3, 2026",
        "type": "Street & Stadium / High Asphalt Temp"
    },
    {
        "round": 5,
        "race_name": "Canadian Grand Prix",
        "circuit_label": "Montreal (Canadian GP)",
        "country": "Canada",
        "flag": "🇨🇦",
        "date": "May 17, 2026",
        "type": "Stop-and-Go / Heavy Braking & Kerb Riding"
    },
    {
        "round": 6,
        "race_name": "Monaco Grand Prix",
        "circuit_label": "Monaco (Monaco GP)",
        "country": "Monaco",
        "flag": "🇲🇨",
        "date": "May 24, 2026",
        "type": "Maximum Downforce / Street Circuit"
    },
    {
        "round": 7,
        "race_name": "Barcelona Grand Prix",
        "circuit_label": "Catalunya (Spanish GP)",
        "country": "Spain",
        "flag": "🇪🇸",
        "date": "June 7, 2026",
        "type": "Chassis Benchmark / Lateral Degradation"
    },
    {
        "round": 8,
        "race_name": "Austrian Grand Prix",
        "circuit_label": "Red Bull Ring (Austrian GP)",
        "country": "Austria",
        "flag": "🇦🇹",
        "date": "June 28, 2026",
        "type": "Short Lap / High Altitude & DRS Trains"
    },
    {
        "round": 9,
        "race_name": "British Grand Prix",
        "circuit_label": "Silverstone (British GP)",
        "country": "United Kingdom",
        "flag": "🇬🇧",
        "date": "July 5, 2026",
        "type": "High Speed Flow / Maggotts-Becketts"
    },
    {
        "round": 10,
        "race_name": "Belgian Grand Prix",
        "circuit_label": "Spa-Francorchamps (Belgian GP)",
        "country": "Belgium",
        "flag": "🇧🇪",
        "date": "July 26, 2026",
        "type": "Elevation / Eau Rouge & Kemmel Straight"
    },
    {
        "round": 11,
        "race_name": "Hungarian Grand Prix",
        "circuit_label": "Hungaroring (Hungarian GP)",
        "country": "Hungary",
        "flag": "🇭🇺",
        "date": "August 2, 2026",
        "type": "Tight & Twisty / Low Overtaking"
    },
    {
        "round": 12,
        "race_name": "Dutch Grand Prix",
        "circuit_label": "Zandvoort (Dutch GP)",
        "country": "Netherlands",
        "flag": "🇳🇱",
        "date": "August 30, 2026",
        "type": "Banked Corners / Coastal Dunes"
    },
    {
        "round": 13,
        "race_name": "Italian Grand Prix",
        "circuit_label": "Monza (Italian GP)",
        "country": "Italy",
        "flag": "🇮🇹",
        "date": "September 6, 2026",
        "type": "Temple of Speed / Minimum Downforce"
    },
    {
        "round": 14,
        "race_name": "Spanish Grand Prix",
        "circuit_label": "Imola (Emilia Romagna GP)",
        "country": "Italy",
        "flag": "🇮🇹",
        "date": "September 13, 2026",
        "type": "Historic Technical Layout"
    },
    {
        "round": 15,
        "race_name": "Azerbaijan Grand Prix",
        "circuit_label": "Baku City (Azerbaijan GP)",
        "country": "Azerbaijan",
        "flag": "🇦🇿",
        "date": "September 20, 2026",
        "type": "Castle Section & 2.2 km Slipstream"
    },
    {
        "round": 16,
        "race_name": "Malaysian Grand Prix",
        "circuit_label": "Sepang (Malaysian GP)",
        "country": "Malaysia",
        "flag": "🇲🇾",
        "date": "September 27, 2026",
        "type": "Tropical Heat & Monsoon / Dual 900m Straights"
    },
    {
        "round": 17,
        "race_name": "Singapore Grand Prix",
        "circuit_label": "Marina Bay (Singapore GP)",
        "country": "Singapore",
        "flag": "🇸🇬",
        "date": "October 4, 2026",
        "type": "Physical Heat / High Tyre Deg Night Race"
    },
    {
        "round": 18,
        "race_name": "United States Grand Prix",
        "circuit_label": "Austin COTA (United States GP)",
        "country": "United States",
        "flag": "🇺🇸",
        "date": "October 18, 2026",
        "type": "Sector 1 Esses / Elevation into T1"
    },
    {
        "round": 19,
        "race_name": "Mexico City Grand Prix",
        "circuit_label": "Autodromo Hermanos Rodriguez (Mexico GP)",
        "country": "Mexico",
        "flag": "🇲🇽",
        "date": "October 25, 2026",
        "type": "2,200m Thin Air / Foro Sol Stadium"
    },
    {
        "round": 20,
        "race_name": "São Paulo Grand Prix",
        "circuit_label": "Interlagos (Sao Paulo GP)",
        "country": "Brazil",
        "flag": "🇧🇷",
        "date": "November 8, 2026",
        "type": "Anti-Clockwise / Weather Drama & Senna S"
    },
    {
        "round": 21,
        "race_name": "Las Vegas Grand Prix",
        "circuit_label": "Las Vegas Strip Circuit",
        "country": "United States",
        "flag": "🇺🇸",
        "date": "November 21, 2026",
        "type": "Cold Night Temp / 350 km/h Strip"
    },
    {
        "round": 22,
        "race_name": "Qatar Grand Prix",
        "circuit_label": "Lusail (Qatar GP)",
        "country": "Qatar",
        "flag": "🇶🇦",
        "date": "November 29, 2026",
        "type": "Medium-High Speed Sweepers"
    },
    {
        "round": 23,
        "race_name": "Saudi Arabian Grand Prix",
        "circuit_label": "Jeddah Corniche (Saudi Arabian GP)",
        "country": "Saudi Arabia",
        "flag": "🇸🇦",
        "date": "December 6, 2026",
        "type": "High Speed Street Circuit"
    },
    {
        "round": 24,
        "race_name": "Abu Dhabi Grand Prix",
        "circuit_label": "Yas Marina (Abu Dhabi GP)",
        "country": "United Arab Emirates",
        "flag": "🇦🇪",
        "date": "December 13, 2026",
        "type": "Season Finale / Day-to-Night Twilight"
    }
]

# Official FIA F1 Points System (P1 to P10)
FIA_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

def simulate_season_until_end(start_round: int = 1, predictor: F1Predictor = None, is_rain_chaos: bool = False):
    """
    Simulates all remaining rounds from start_round to Round 24 (Season Finale)
    using the ML Predictor, circuit profiles, and dynamic championship momentum.
    """
    if predictor is None:
        predictor = F1Predictor()
        predictor.load()

    # Initialize standings dictionary
    driver_stats = {}
    for d in CURRENT_ROSTER:
        driver_stats[d["driver"]] = {
            "driver": d["driver"],
            "name": d["name"],
            "team": d["team"],
            "points": 0,
            "wins": 0,
            "podiums": 0,
            "top10s": 0,
            "base_pace": d["base_pace"],
            "recent_positions": [max(1, int(11.0 - d["base_pace"]))] * 5
        }

    team_points = {}
    for d in CURRENT_ROSTER:
        team_points[d["team"]] = 0

    round_forecasts = []
    trajectory_records = []

    # Filter to remaining rounds
    remaining_rounds = [r for r in SEASON_2026_CALENDAR if r["round"] >= start_round]

    for round_info in remaining_rounds:
        r_num = round_info["round"]
        c_label = round_info["circuit_label"]
        c_info = CIRCUITS.get(c_label, list(CIRCUITS.values())[0])
        df_idx = c_info.get("downforce_index", 0.6)

        # 1. Qualifying Simulation based on car aeropack affinity & driver pace
        qualifying_scores = []
        for d in CURRENT_ROSTER:
            code = d["driver"]
            stats = driver_stats[code]
            
            # Circuit aerodynamic affinity
            aero_affinity = 0.0
            if df_idx >= 0.70:  # High Downforce (Monaco, Hungaroring, Singapore, Zandvoort)
                if d["team"] in ["McLaren", "Ferrari"]:
                    aero_affinity = 0.40
            elif df_idx <= 0.45:  # Low Downforce / Power (Monza, Spa, Baku, Vegas)
                if d["team"] in ["Red Bull Racing", "Mercedes"]:
                    aero_affinity = 0.38

            # Rolling form factor (average finish over last 3 races)
            recent_avg = np.mean(stats["recent_positions"][-3:])
            form_bonus = max(-0.25, min(0.25, (10.0 - recent_avg) * 0.04))

            # Small race-to-race variance
            variance = np.random.normal(0.0, 0.12)
            round_pace = stats["base_pace"] + aero_affinity + form_bonus + variance
            qualifying_scores.append((code, round_pace))

        qualifying_scores.sort(key=lambda x: x[1], reverse=True)

        # Build starting grid for this round
        grid = []
        for q_pos, (d_code, _) in enumerate(qualifying_scores, start=1):
            stats = driver_stats[d_code]
            grid.append({
                "driver": d_code,
                "name": stats["name"],
                "team": stats["team"],
                "grid_position": q_pos,
                "driver_rolling_avg_pos": float(np.mean(stats["recent_positions"])),
                "team_rolling_pts": float(team_points[stats["team"]])
            })

        # Rain condition for this round
        round_rain = is_rain_chaos or (round_info.get("country") in ["United Kingdom", "Belgium", "Brazil"] and np.random.rand() < 0.25)

        # 2. Execute ML Race Inference
        race_pred = predictor.predict_race(
            starting_grid=grid,
            circuit_info=c_info,
            is_rain=round_rain
        )

        p1_res = race_pred.iloc[0]
        p2_res = race_pred.iloc[1]
        p3_res = race_pred.iloc[2]

        # 3. Award FIA Championship Points
        for pos_idx, pts in enumerate(FIA_POINTS):
            if pos_idx < len(race_pred):
                car = race_pred.iloc[pos_idx]
                code = car["driver"]
                driver_stats[code]["points"] += pts
                driver_stats[code]["top10s"] += 1
                team_points[car["team"]] += pts

        # Award 1 point for fastest lap to a top 10 finisher
        fastest_lap_driver = p1_res["driver"] if np.random.rand() > 0.35 else p2_res["driver"]
        driver_stats[fastest_lap_driver]["points"] += 1

        # Track wins & podiums
        driver_stats[p1_res["driver"]]["wins"] += 1
        driver_stats[p1_res["driver"]]["podiums"] += 1
        driver_stats[p2_res["driver"]]["podiums"] += 1
        driver_stats[p3_res["driver"]]["podiums"] += 1

        # Update rolling finish history
        for pos_idx, row in race_pred.iterrows():
            code = row["driver"]
            driver_stats[code]["recent_positions"].append(pos_idx + 1)
            if len(driver_stats[code]["recent_positions"]) > 5:
                driver_stats[code]["recent_positions"].pop(0)

        # Record round forecast summary
        downforce_badge = "High Downforce" if df_idx >= 0.70 else ("Low Downforce / Speed" if df_idx <= 0.45 else "Medium Aero")
        strat_key = "1-Stop Endurance" if c_info.get("tire_degradation") != "High" else "Aggressive 2-Stop (Undercut)"
        if round_rain:
            strat_key = "🌧️ Wet / Intermediate Crossover"

        round_forecasts.append({
            "Round": r_num,
            "Flag": round_info["flag"],
            "Grand Prix": round_info["race_name"],
            "Circuit": round_info["circuit_label"],
            "Projected Winner (P1)": f"{p1_res['driver']} ({p1_res['team']})",
            "Win Probability": f"{p1_res['win_prob']*100:.1f}%",
            "Projected P2": f"{p2_res['driver']} ({p2_res['team']})",
            "Projected P3": f"{p3_res['driver']} ({p3_res['team']})",
            "Aero Profile": downforce_badge,
            "Strategy Outlook": strat_key,
            "raw_winner": p1_res['driver'],
            "raw_team": p1_res['team'],
            "raw_win_prob": p1_res['win_prob']
        })

        # Save trajectory snapshot for chart
        snapshot = {"Round": f"R{r_num}: {round_info['race_name'].replace(' Grand Prix', '')}"}
        for code, stats in driver_stats.items():
            snapshot[code] = stats["points"]
        trajectory_records.append(snapshot)

    # Compile WDC Standings Table
    wdc_list = list(driver_stats.values())
    wdc_df = pd.DataFrame(wdc_list).sort_values(by=["points", "wins", "podiums"], ascending=[False, False, False]).reset_index(drop=True)
    wdc_df["Rank"] = range(1, len(wdc_df) + 1)

    # Compile WCC Standings Table
    wcc_records = []
    teams = set([d["team"] for d in CURRENT_ROSTER])
    for t in teams:
        t_drivers = [d for d in wdc_list if d["team"] == t]
        t_pts = sum(d["points"] for d in t_drivers)
        t_wins = sum(d["wins"] for d in t_drivers)
        t_pods = sum(d["podiums"] for d in t_drivers)
        wcc_records.append({
            "Team": t,
            "Points": t_pts,
            "Wins": t_wins,
            "Podiums": t_pods
        })
    wcc_df = pd.DataFrame(wcc_records).sort_values(by=["Points", "Wins"], ascending=[False, False]).reset_index(drop=True)
    wcc_df["Rank"] = range(1, len(wcc_df) + 1)

    forecast_df = pd.DataFrame(round_forecasts)
    traj_df = pd.DataFrame(trajectory_records)

    return {
        "calendar_forecast_df": forecast_df,
        "wdc_df": wdc_df,
        "wcc_df": wcc_df,
        "trajectory_df": traj_df,
        "champion_driver": wdc_df.iloc[0],
        "champion_constructor": wcc_df.iloc[0]
    }

def create_championship_trajectory_chart(trajectory_df: pd.DataFrame, top_n: int = 6):
    """Generates an interactive Plotly trajectory chart showing points accumulation until the season finale."""
    fig = go.Figure()

    if trajectory_df.empty:
        return fig

    last_row = trajectory_df.iloc[-1]
    driver_codes = [col for col in trajectory_df.columns if col != "Round"]
    sorted_drivers = sorted(driver_codes, key=lambda c: last_row[c], reverse=True)[:top_n]

    # Map driver to team color
    driver_to_team = {d["driver"]: d["team"] for d in CURRENT_ROSTER}

    for d_code in sorted_drivers:
        team_name = driver_to_team.get(d_code, "Red Bull Racing")
        color = TEAM_COLORS.get(team_name, "#FFFFFF")
        
        fig.add_trace(go.Scatter(
            x=trajectory_df["Round"],
            y=trajectory_df[d_code],
            mode="lines+markers",
            name=f"{d_code} ({team_name})",
            line=dict(color=color, width=3),
            marker=dict(size=6),
            hovertemplate=f"<b>{d_code}</b><br>Points: %{{y}}<extra></extra>"
        ))

    fig.update_layout(
        title="<b>2026 World Drivers' Championship Points Trajectory (Until Abu Dhabi Finale)</b>",
        xaxis=dict(title="", tickangle=-45, showgrid=True, gridcolor="#22272E"),
        yaxis=dict(title="Cumulative Championship Points", showgrid=True, gridcolor="#22272E"),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        height=480,
        margin=dict(l=30, r=30, t=60, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

def create_win_distribution_chart(forecast_df: pd.DataFrame):
    """Creates a donut chart of projected race wins across remaining rounds."""
    if forecast_df.empty or "raw_winner" not in forecast_df.columns:
        return go.Figure()

    win_counts = forecast_df["raw_winner"].value_counts().reset_index()
    win_counts.columns = ["Driver", "Wins"]

    driver_to_team = {d["driver"]: d["team"] for d in CURRENT_ROSTER}
    colors = [TEAM_COLORS.get(driver_to_team.get(d, ""), "#FFFFFF") for d in win_counts["Driver"]]

    fig = go.Figure(data=[go.Pie(
        labels=win_counts["Driver"],
        values=win_counts["Wins"],
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#161B22", width=2)),
        textinfo="label+value",
        hoverinfo="label+value+percent"
    )])

    fig.update_layout(
        title="<b>Projected Wins Distribution (Remaining Rounds)</b>",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True
    )
    return fig

_CACHED_2026_EVAL = None
_CACHED_2026_MTIME = 0

def evaluate_season_2026_accuracy(predictor=None):
    """
    Evaluates AI prediction accuracy against real 2026 season data (Rounds 1–21).
    Computes exact winner hit rate, podium overlap precision, and position MAE.
    """
    global _CACHED_2026_EVAL, _CACHED_2026_MTIME

    import os
    from src.data_engine import get_latest_dataset, HIST_DATA_FILE
    from src.circuits import get_circuit_by_race_name

    mtime = os.path.getmtime(HIST_DATA_FILE) if os.path.exists(HIST_DATA_FILE) else 0
    if _CACHED_2026_EVAL is not None and mtime == _CACHED_2026_MTIME:
        return _CACHED_2026_EVAL

    if predictor is None:
        from src.model_engine import F1Predictor
        predictor = F1Predictor()
        predictor.load()

    df = get_latest_dataset()
    df_2026 = df[df["season"] == 2026]

    if df_2026.empty:
        return {"summary": {}, "rounds_df": pd.DataFrame()}

    cal_map = {item["round"]: item for item in SEASON_2026_CALENDAR}

    rounds_records = []
    total_winner_hits = 0
    total_pod_hits = 0
    total_mae = 0.0

    for r in sorted(df_2026["round"].unique()):
        sub = df_2026[df_2026["round"] == r]
        c_name = sub["circuit"].iloc[0]
        c_label, c_info = get_circuit_by_race_name(c_name)
        cal_meta = cal_map.get(r, {"flag": "🏁", "race_name": c_name})

        grid = []
        for _, row in sub.sort_values("grid_position").iterrows():
            grid.append({
                "driver": row["driver"],
                "team": row["team"],
                "grid_position": int(row["grid_position"]),
                "driver_rolling_avg_pos": float(row["driver_rolling_avg_pos"]),
                "team_rolling_pts": float(row["team_rolling_pts"]),
                "actual_final_position": int(row["final_position"])
            })

        pred_df = predictor.predict_race(
            starting_grid=grid,
            circuit_info=c_info,
            is_rain=(sub["rain_prob"].iloc[0] > 0.5)
        )
        act_map = {item["driver"]: item["actual_final_position"] for item in grid}
        pred_df["actual_final_position"] = pred_df["driver"].map(act_map)

        actual_p1 = sub[sub["final_position"] == 1].iloc[0]
        actual_p2 = sub[sub["final_position"] == 2].iloc[0]
        actual_p3 = sub[sub["final_position"] == 3].iloc[0]

        pred_p1 = pred_df.iloc[0]
        pred_p2 = pred_df.iloc[1]
        pred_p3 = pred_df.iloc[2]

        is_winner_hit = (actual_p1["driver"] == pred_p1["driver"])
        if is_winner_hit:
            total_winner_hits += 1

        act_pod_set = set(sub[sub["final_position"] <= 3]["driver"])
        pred_pod_set = set(pred_df.iloc[:3]["driver"])
        pod_hits = len(act_pod_set.intersection(pred_pod_set))
        total_pod_hits += pod_hits

        mae = float((pred_df["predicted_rank"] - pred_df["actual_final_position"]).abs().mean())
        total_mae += mae

        rounds_records.append({
            "Round": f"R{r:02d}",
            "Round_Num": r,
            "Flag": cal_meta.get("flag", "🏁"),
            "Grand Prix": cal_meta.get("race_name", c_name),
            "Circuit": c_label,
            "Actual Winner (P1)": f"{actual_p1['driver']} ({actual_p1['team']})",
            "AI Projected Winner": f"{pred_p1['driver']} ({pred_p1['team']})",
            "Winner Match": "🎯 HIT" if is_winner_hit else "❌ MISSED",
            "Podium Overlap": f"{pod_hits}/3 ({pod_hits/3*100:.0f}%)",
            "pod_hits_num": pod_hits,
            "winner_hit_bool": is_winner_hit,
            "Position Error (MAE)": f"±{mae:.2f} pos",
            "mae_val": mae,
            "Actual Podium": f"P1 {actual_p1['driver']} | P2 {actual_p2['driver']} | P3 {actual_p3['driver']}",
            "AI Predicted Podium": f"P1 {pred_p1['driver']} | P2 {pred_p2['driver']} | P3 {pred_p3['driver']}"
        })

    n = len(rounds_records)
    summary = {
        "total_races": n,
        "winner_hits": total_winner_hits,
        "winner_accuracy_pct": round(total_winner_hits / n * 100, 1) if n > 0 else 0,
        "podium_accuracy_pct": round(total_pod_hits / (n * 3) * 100, 1) if n > 0 else 0,
        "avg_mae": round(total_mae / n, 2) if n > 0 else 0,
        "latest_round": n,
        "latest_grand_prix": rounds_records[-1]["Grand Prix"] if n > 0 else "N/A",
        "latest_actual_winner": rounds_records[-1]["Actual Winner (P1)"] if n > 0 else "N/A",
        "latest_podium": rounds_records[-1]["Actual Podium"] if n > 0 else "N/A"
    }

    _CACHED_2026_EVAL = {
        "summary": summary,
        "rounds_df": pd.DataFrame(rounds_records)
    }
    return _CACHED_2026_EVAL

def create_season_2026_accuracy_chart(rounds_df: pd.DataFrame):
    """Generates dual-axis interactive Plotly chart of 2026 real results vs AI accuracy."""
    if rounds_df.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_colors = ["#00D2BE" if row["winner_hit_bool"] else "#E10600" for _, row in rounds_df.iterrows()]

    fig.add_trace(go.Bar(
        x=rounds_df["Round"],
        y=rounds_df["pod_hits_num"],
        name="Podium Hits (out of 3)",
        marker=dict(color=bar_colors, line=dict(color="#FFFFFF", width=1)),
        text=[f"{row['Winner Match']}<br>{row['Podium Overlap']}" for _, row in rounds_df.iterrows()],
        textposition="inside",
        hovertext=[
            f"<b>{row['Grand Prix']}</b> ({row['Round']})<br>"
            f"Actual Winner: {row['Actual Winner (P1)']}<br>"
            f"AI Winner: {row['AI Projected Winner']}<br>"
            f"Podium Overlap: {row['Podium Overlap']}"
            for _, row in rounds_df.iterrows()
        ],
        hoverinfo="text",
        opacity=0.85
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=rounds_df["Round"],
        y=rounds_df["mae_val"],
        name="Position Error (MAE)",
        mode="lines+markers",
        line=dict(color="#FFE500", width=3),
        marker=dict(size=8, color="#FFE500", symbol="diamond"),
        hovertext=[f"MAE: ±{row['mae_val']:.2f} positions" for _, row in rounds_df.iterrows()],
        hoverinfo="text+name"
    ), secondary_y=True)

    fig.update_layout(
        title="<b>2026 Season Real-World Accuracy: Winner & Podium Hits vs Grid Deviation (Rounds 1–21)</b>",
        xaxis=dict(title="2026 Championship Round", showgrid=False),
        yaxis=dict(title="Podium Finishers Matched (0 to 3)", range=[0, 3.5], showgrid=True, gridcolor="#22272E"),
        yaxis2=dict(title="Position Error (MAE, lower is better)", range=[0, 8], showgrid=False),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        height=420,
        margin=dict(l=30, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

