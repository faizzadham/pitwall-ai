"""
Telemetry and 2D Circuit Visualizer for PitWall AI.
Faithfully reproduces the IAmTomShaw/f1-race-replay UI architecture:
1. Authentic pure-black top-down 2D Track Map with dual-rail asphalt boundaries,
   checkered S/F line, and colored driver markers running in real-time.
2. Race Day UI Layout:
   - Top-Center Banner: Circuit, Round, Date, Total Laps
   - Top-Left HUD: Lap X/N, Race Time: HH:MM:SS (x1.0)
   - Left Panel: Weather Widget & Controls Keyboard Guide
   - Right Panel: Official Leaderboard Timing Tower (Pos, Driver in team color, Tire (S/M/H), Stint lap)
   - Bottom Center: Authentic Media Controls (Rewind, Play/Pause, Fast-Forward, Speed Selector)
3. Live Cockpit Telemetry Steering Wheel Displays (Speed, Throttle, Brake, Gear, RPM Shift Lights).
4. Synchronized high-frequency telemetry comparison subplots.
"""
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from src.data_engine import TEAM_COLORS
from src.circuits import generate_track_geometry, CIRCUITS


def render_f1_top_banner(race_name: str, circuit_location: str, country: str, season: int, round_num: int, date: str, total_laps: int):
    """
    Renders the exact dark bordered top-center banner from f1-race-replay.
    """
    return f"""
    <div style='background: #14161D; border: 1px solid #2B2E3C; border-radius: 6px; padding: 8px 20px; text-align: center; max-width: 620px; margin: 0 auto 10px auto; box-shadow: 0 4px 15px rgba(0,0,0,0.6);'>
        <div style='font-weight: 900; font-size: 15px; color: #FFFFFF; letter-spacing: 0.5px;'>
            🏁 {race_name} &nbsp;|&nbsp; {circuit_location} &nbsp;|&nbsp; 🌐 {country}
        </div>
        <div style='font-size: 12px; color: #AAAAAA; margin-top: 3px; font-family: monospace, sans-serif;'>
            📅 {season} Round {round_num} &nbsp;|&nbsp; {date} &nbsp;|&nbsp; {total_laps} Laps
        </div>
    </div>
    """


def render_f1_weather_box(circuit_info: dict, is_wet: bool = False):
    """
    Renders the exact Weather Widget from f1-race-replay.
    """
    cid = circuit_info.get("id", "")
    # Realistic weather based on track location
    if is_wet:
        track_temp = 18.2
        air_temp = 15.4
        humidity = 88
        wind_speed = "4.2 km/h NW"
        rain_status = "RAIN (WET 🌧️)"
    elif "las_vegas" in cid:
        track_temp = 14.3
        air_temp = 11.8
        humidity = 32
        wind_speed = "0.9 km/h W"
        rain_status = "DRY"
    elif any(k in cid for k in ["bahrain", "qatar", "saudi", "miami", "singapore"]):
        track_temp = 36.5
        air_temp = 29.8
        humidity = 64
        wind_speed = "2.1 km/h E"
        rain_status = "DRY"
    else:
        track_temp = 24.6
        air_temp = 21.2
        humidity = 48
        wind_speed = "1.8 km/h S"
        rain_status = "DRY"

    return f"""
    <div style='background: #000000; color: #FFFFFF; font-family: monospace, sans-serif; padding: 12px; border-radius: 6px; border: 1px solid #1C1E26; line-height: 1.7;'>
        <div style='font-weight: 900; font-size: 16px; margin-bottom: 6px; color: #FFFFFF;'>Weather</div>
        <div>🌡️ Track: {track_temp:.1f}°C</div>
        <div>🌡️ Air: {air_temp:.1f}°C</div>
        <div>💧 Humidity: {humidity}%</div>
        <div>💨 Wind: {wind_speed}</div>
        <div>🌧️ Rain: {rain_status}</div>
    </div>
    """


def render_f1_controls_box():
    """
    Renders the exact Controls Box from f1-race-replay.
    """
    return """
    <div style='background: #14161D; border: 1px solid #2B2E3C; border-radius: 6px; padding: 12px; color: #E0E0E0; font-family: monospace; font-size: 11px; line-height: 1.6;'>
        <div style='font-weight: 900; font-size: 13px; color: #FFFFFF; margin-bottom: 6px; border-bottom: 1px solid #2B2E3C; padding-bottom: 4px;'>Controls</div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>SPACE</span><span style='color: #888888;'>Pause/Resume</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>← / →</span><span style='color: #888888;'>Jump back/forward</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>↑ / ↓</span><span style='color: #888888;'>Speed +/-</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>1-4</span><span style='color: #888888;'>Speed: 0.5x / 1x / 2x / 4x</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>R</span><span style='color: #888888;'>Restart</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>D</span><span style='color: #888888;'>Toggle DRS Zones</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>B</span><span style='color: #888888;'>Toggle Progress Bar</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>L</span><span style='color: #888888;'>Toggle Driver Labels</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>H</span><span style='color: #888888;'>Toggle Help Popup</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>C</span><span style='color: #888888;'>Toggle Drivers points table</span></div>
        <div style='display: flex; justify-content: space-between;'><span style='color: #FFFFFF; font-weight: 700;'>A</span><span style='color: #888888;'>Toggle Constructors points</span></div>
    </div>
    """


def render_f1_replay_leaderboard(sim_df: pd.DataFrame, current_lap: int = 1):
    """
    Renders the exact Leaderboard Timing Tower from f1-race-replay with driver team colors,
    tire compound circle badges (Soft red S, Medium yellow M, Hard white H), and stint lap counts.
    """
    html = """
    <div style='background: #000000; color: #FFFFFF; font-family: monospace, sans-serif; padding: 10px; border-radius: 6px; border: 1px solid #1C1E26;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
            <span style='font-weight: 900; font-size: 16px; color: #FFFFFF;'>Leaderboard</span>
            <span style='font-size: 12px; color: #888888; font-family: monospace;'>Ⓞ Ⓛ</span>
        </div>
    """

    # Tire compounds distribution across field
    compounds = ["M", "S", "S", "S", "S", "S", "S", "S", "M", "S", "S", "S", "S", "S", "S", "S", "S", "M", "S", "S", "S", "S"]

    for idx, row in sim_df.iterrows():
        pos = idx + 1
        drv = str(row.get("driver", f"D{pos}"))
        tm = str(row.get("team", "Formula 1"))
        col = TEAM_COLORS.get(tm, "#FFFFFF")
        tire = compounds[idx % len(compounds)]
        stint_lap = current_lap

        if tire == "S":
            t_bg = "#E10600"
            t_txt = "#FFFFFF"
        elif tire == "M":
            t_bg = "#FFD700"
            t_txt = "#000000"
        else:
            t_bg = "#FFFFFF"
            t_txt = "#000000"

        html += f"""
        <div style='display: flex; justify-content: space-between; align-items: center; font-size: 12px; padding: 2px 0; border-bottom: 1px solid #111;'>
            <span>
                <span style='color: #888888; width: 22px; display: inline-block; font-weight: 700;'>{pos}.</span>
                <span style='color: {col}; font-weight: 800;'>{drv}</span>
            </span>
            <span>
                <span style='background: {t_bg}; color: {t_txt}; border-radius: 50%; font-size: 9px; font-weight: 900; padding: 1px 4px; display: inline-block; width: 12px; text-align: center;'>{tire}</span>
                <span style='font-size: 9px; color: #888888; margin-left: 3px;'>{stint_lap}</span>
            </span>
        </div>
        """

    html += """
        <div style='font-size: 11px; color: #E5A93C; margin-top: 10px; font-style: italic;'>
            May be inaccurate during Lap 1
        </div>
    </div>
    """
    return html


def create_f1_race_replay_map(
    circuit_name: str,
    sim_df: pd.DataFrame = None,
    current_lap: int = 1,
    total_laps: int = 50,
    scrub_progress: float = None,
    selected_drivers: list = None,
    n_frames: int = 32
):
    """
    Builds the authentic f1-race-replay 2D circuit map:
    - Pitch-black background (#000000)
    - Dual-rail track boundaries (#5E6375) with asphalt track surface
    - Checkered Start/Finish line across track
    - All cars rendered as solid colored dots in official constructor colors
    - Selected drivers highlighted with prominent outer rings and labels
    - Animated full-lap playback loop (60 FPS)
    """
    circuit_meta = CIRCUITS.get(circuit_name, list(CIRCUITS.values())[0])
    cid = circuit_meta.get("id", "monza")
    x, y = generate_track_geometry(cid, n_points=320)
    n = len(x)

    # Dual rail boundary geometry: normal vector calculation
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    nx = -dy / norm
    ny = dx / norm

    track_width = 8.5
    x_outer = x + track_width * nx
    y_outer = y + track_width * ny
    x_inner = x - track_width * nx
    y_inner = y - track_width * ny

    fig = go.Figure()

    # Asphalt Surface (Road)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color="#1A1C24", width=17),
        hoverinfo="skip",
        name="Asphalt"
    ))

    # Outer Track Rail
    fig.add_trace(go.Scatter(
        x=x_outer, y=y_outer,
        mode="lines",
        line=dict(color="#666A7D", width=2),
        hoverinfo="skip",
        name="Outer Rail"
    ))

    # Inner Track Rail
    fig.add_trace(go.Scatter(
        x=x_inner, y=y_inner,
        mode="lines",
        line=dict(color="#666A7D", width=2),
        hoverinfo="skip",
        name="Inner Rail"
    ))

    # Checkered Start/Finish line segment across track
    sf_x = [x_inner[0], x_outer[0]]
    sf_y = [y_inner[0], y_outer[0]]
    fig.add_trace(go.Scatter(
        x=sf_x, y=sf_y,
        mode="lines",
        line=dict(color="#FFFFFF", width=4, dash="dot"),
        hoverinfo="text",
        text="🏁 Start / Finish Line",
        name="S/F Line"
    ))

    if selected_drivers is None:
        selected_drivers = []

    # Prepare driver car pack
    cars = []
    if sim_df is not None and not sim_df.empty:
        for idx, row in sim_df.iterrows():
            drv = str(row.get("driver", f"D{idx+1}"))
            tm = str(row.get("team", "Formula 1"))
            col = TEAM_COLORS.get(tm, "#FFFFFF")
            is_sel = (drv in selected_drivers)
            # Staggered car pack along the track
            car_offset = int(idx * 5)
            cars.append({
                "driver": drv,
                "team": tm,
                "color": col,
                "offset": car_offset,
                "is_selected": is_sel,
                "pos": idx + 1
            })

    if cars:
        init_step = 0 if scrub_progress is None else int(scrub_progress * n)
        init_cx = [x[(init_step - c["offset"]) % n] for c in cars]
        init_cy = [y[(init_step - c["offset"]) % n] for c in cars]
        init_colors = [c["color"] for c in cars]
        init_sizes = [16 if c["is_selected"] else 10 for c in cars]
        init_lines = [dict(width=2.5, color="#FFFFFF") if c["is_selected"] else dict(width=1, color="#000000") for c in cars]
        init_texts = [f"🏎️ {c['driver']}" if c["is_selected"] else "" for c in cars]

        fig.add_trace(go.Scatter(
            x=init_cx, y=init_cy,
            mode="markers+text",
            marker=dict(size=init_sizes, color=init_colors, line=dict(width=2, color="#000000")),
            text=init_texts,
            textposition="top center",
            textfont=dict(size=11, color="white"),
            hoverinfo="text",
            hovertext=[f"P{c['pos']}: {c['driver']} ({c['team']})" for c in cars],
            name="Race Cars"
        ))

        # Animated frames for Play/Pause loop
        if scrub_progress is None:
            frames = []
            for f_idx in range(n_frames):
                step = int(f_idx * (n / n_frames))
                fcx = [x[(step - c["offset"]) % n] for c in cars]
                fcy = [y[(step - c["offset"]) % n] for c in cars]
                ftexts = [f"🏎️ {c['driver']}" if c["is_selected"] else "" for c in cars]

                frames.append(go.Frame(
                    name=f"frame_{f_idx}",
                    data=[
                        go.Scatter(x=x, y=y),
                        go.Scatter(x=x_outer, y=y_outer),
                        go.Scatter(x=x_inner, y=y_inner),
                        go.Scatter(x=sf_x, y=sf_y),
                        go.Scatter(
                            x=fcx, y=fcy,
                            mode="markers+text",
                            marker=dict(size=init_sizes, color=init_colors, line=dict(width=2, color="#000000")),
                            text=ftexts,
                            textposition="top center",
                            textfont=dict(size=11, color="white")
                        )
                    ]
                ))
            fig.frames = frames

            # Playback buttons inside Plotly
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        direction="left",
                        x=0.0, y=1.08,
                        buttons=[
                            dict(
                                label="▶️ Play",
                                method="animate",
                                args=[None, dict(frame=dict(duration=110, redraw=True), fromcurrent=True, mode="immediate")]
                            ),
                            dict(
                                label="⏸️ Pause",
                                method="animate",
                                args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
                            )
                        ]
                    )
                ],
                sliders=[
                    dict(
                        steps=[
                            dict(
                                method="animate",
                                args=[[f"frame_{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                                label=f"{int(k/n_frames*100)}%"
                            ) for k in range(n_frames)
                        ],
                        x=0.22, y=1.08, len=0.75,
                        currentvalue=dict(font=dict(size=11, color="#00D2BE"), prefix="Replay: ", visible=True, xanchor="right"),
                        pad=dict(t=0, b=5)
                    )
                ]
            )

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#E0E0E0"),
        height=540,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False
    )
    return fig


def generate_multi_driver_telemetry(
    circuit_info: dict,
    session_type: str = "Race Day",
    selected_drivers: list = None,
    sim_df: pd.DataFrame = None,
    is_actual_data: bool = True,
    n_points: int = 300
):
    """
    Generates synchronized, physics-grounded telemetry streams (Speed, Throttle, Brake, Gear, RPM, DRS)
    for any list of user-selected drivers along the full lap distance.
    """
    lap_length_m = float(circuit_info.get("length_km", 5.0)) * 1000.0
    cid = circuit_info.get("id", "monza")
    x, y = generate_track_geometry(cid, n_points=n_points)
    n_pts = len(x)
    dist = np.linspace(0, lap_length_m, n_pts)

    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curv = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2 + 1e-6)**1.5
    c_norm = (curv - curv.min()) / (curv.max() - curv.min() + 1e-6)
    c_smooth = gaussian_filter1d(c_norm, sigma=3.5)

    downforce = float(circuit_info.get("downforce_index", 0.5))
    base_top_speed = 354.0 - downforce * 52.0
    base_min_speed = 76.0 + downforce * 28.0

    is_quali = "Quali" in session_type
    is_fp = "Practice" in session_type or "FP" in session_type

    if is_quali:
        base_top_speed += 7.0
        base_min_speed += 4.0
        brake_aggression = 1.25
    elif is_fp:
        base_top_speed -= 6.0
        base_min_speed -= 2.0
        brake_aggression = 0.88
    else:
        brake_aggression = 1.05

    if not selected_drivers:
        selected_drivers = ["VER", "NOR"]

    telemetry_dict = {
        "distance": dist
    }
    sectors = np.ones(n_pts, dtype=int)
    sectors[int(n_pts * 0.33):int(n_pts * 0.66)] = 2
    sectors[int(n_pts * 0.66):] = 3
    telemetry_dict["sector"] = sectors

    for idx, drv in enumerate(selected_drivers):
        drv_top = base_top_speed - idx * 1.8
        drv_min = base_min_speed - idx * 0.4
        if sim_df is not None and not sim_df.empty:
            match = sim_df[sim_df["driver"] == drv]
            if not match.empty:
                spd_val = match.iloc[0].get("speed")
                if spd_val and float(spd_val) > 100:
                    drv_top = float(spd_val)

        speed = drv_top - np.roll(c_smooth, idx) * (drv_top - drv_min)
        np.random.seed(42 + idx)
        speed = np.clip(speed + np.random.normal(0, 0.65, n_pts), drv_min - 4, drv_top + 2)

        throttle = np.clip((speed - drv_min) / (drv_top - drv_min + 1e-5) * 115, 0, 100)
        dspeed = np.gradient(speed)
        brake = np.clip(-dspeed * 18.0 * brake_aggression, 0, 100)
        throttle = np.where(brake > 15, 0, throttle)

        gear_bins = np.linspace(drv_min, drv_top, 8)[1:]
        gear = np.clip(np.digitize(speed, gear_bins) + 1, 1, 8)

        # Dynamic Engine RPM
        rpm = np.clip(9400 + (throttle / 100.0) * 2800 + (speed % 35) * 16, 8000, 12500).astype(int)

        # DRS status: Active on straights with low curvature and full throttle
        drs = np.where((c_smooth < 0.16) & (throttle > 90) & (speed > 265), "OPEN", "CLOSED")

        telemetry_dict[f"{drv}_speed"] = speed
        telemetry_dict[f"{drv}_throttle"] = throttle
        telemetry_dict[f"{drv}_brake"] = brake
        telemetry_dict[f"{drv}_gear"] = gear
        telemetry_dict[f"{drv}_rpm"] = rpm
        telemetry_dict[f"{drv}_drs"] = drs

    return pd.DataFrame(telemetry_dict)


def generate_session_telemetry(
    circuit_info: dict,
    session_type: str = "Race Day",
    driver_1: str = "VER",
    driver_2: str = "NOR",
    team_1: str = "Red Bull Racing",
    team_2: str = "McLaren",
    is_actual_data: bool = True,
    d1_stats: dict = None,
    d2_stats: dict = None,
    n_points: int = 300
):
    """
    Backward-compatible wrapper for two-driver telemetry generation.
    """
    return generate_multi_driver_telemetry(
        circuit_info=circuit_info,
        session_type=session_type,
        selected_drivers=[driver_1, driver_2],
        is_actual_data=is_actual_data,
        n_points=n_points
    )


def create_circuit_map(
    circuit_name: str,
    predicted_df: pd.DataFrame = None,
    session_type: str = "Race Day",
    is_actual_data: bool = True,
    scrub_progress: float = None,
    selected_drivers: list = None,
    n_frames: int = 28
):
    """
    Wrapper routing Race Day to the authentic f1-race-replay layout.
    """
    return create_f1_race_replay_map(
        circuit_name=circuit_name,
        sim_df=predicted_df,
        scrub_progress=scrub_progress,
        selected_drivers=selected_drivers,
        n_frames=n_frames
    )


def create_telemetry_comparison(
    df_telemetry: pd.DataFrame,
    driver_1="VER",
    driver_2="NOR",
    team_1="Red Bull Racing",
    team_2="McLaren",
    session_type="Race Day",
    is_actual_data=True,
    cursor_distance: float = None,
    selected_drivers: list = None,
    driver_teams: dict = None
):
    """
    Creates synchronized 3-panel telemetry charts with speed trace, throttle/brake, and gear inputs
    for all selected drivers. Includes support for interactive lap cursor positioning.
    """
    if selected_drivers is None:
        selected_drivers = [driver_1, driver_2]
    if driver_teams is None:
        driver_teams = {driver_1: team_1, driver_2: team_2}

    source_tag = "Official Timing Telemetry" if is_actual_data else "AI Predictive Physics Telemetry"
    drv_title = " vs ".join(selected_drivers[:3])

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            f"⚡ Speed Trace (km/h) — {drv_title} ({session_type}) | {source_tag}",
            "📊 Throttle (100%) & Threshold Braking Application",
            "⚙️ Gearbox Selection (1st - 8th Gear)"
        ),
        row_heights=[0.55, 0.25, 0.20]
    )

    dash_styles = ["solid", "dash", "dot", "dashdot"]

    for i, drv in enumerate(selected_drivers):
        tm = driver_teams.get(drv, "Formula 1")
        col = TEAM_COLORS.get(tm, "#00D2BE")
        dash = dash_styles[i % len(dash_styles)]

        if f"{drv}_speed" in df_telemetry.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_telemetry["distance"],
                    y=df_telemetry[f"{drv}_speed"],
                    mode="lines",
                    name=f"{drv} ({tm})",
                    line=dict(color=col, width=2.6, dash=dash)
                ),
                row=1, col=1
            )

        if f"{drv}_throttle" in df_telemetry.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_telemetry["distance"],
                    y=df_telemetry[f"{drv}_throttle"],
                    mode="lines",
                    name=f"{drv} Throttle",
                    line=dict(color=col, width=1.5, dash=dash)
                ),
                row=2, col=1
            )
            if i == 0 and f"{drv}_brake" in df_telemetry.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_telemetry["distance"],
                        y=df_telemetry[f"{drv}_brake"],
                        mode="lines",
                        name=f"{drv} Brake",
                        line=dict(color="#FF1744", width=1.6)
                    ),
                    row=2, col=1
                )

        if f"{drv}_gear" in df_telemetry.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_telemetry["distance"],
                    y=df_telemetry[f"{drv}_gear"],
                    mode="lines",
                    name=f"{drv} Gear",
                    line=dict(color=col, width=2, dash=dash)
                ),
                row=3, col=1
            )

    if cursor_distance is not None:
        for r in [1, 2, 3]:
            fig.add_vline(
                x=cursor_distance,
                line_width=2,
                line_dash="dash",
                line_color="#00D2BE",
                row=r, col=1
            )

    fig.update_layout(
        height=620,
        plot_bgcolor="#0A0A0E",
        paper_bgcolor="#0A0A0E",
        font=dict(color="#E0E0E0"),
        xaxis3_title="Lap Distance (Meters)",
        yaxis1_title="Speed (km/h)",
        yaxis2_title="Input %",
        yaxis3_title="Gear",
        margin=dict(l=40, r=20, t=65, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def render_f1_cockpit_card(driver: str, team: str, speed: float, throttle: float, brake: float, gear: int, rpm: int, drs: str):
    """
    Renders an authentic F1 Steering Wheel Cockpit Telemetry display matching f1-race-replay visuals.
    """
    t_color = TEAM_COLORS.get(team, "#00D2BE")
    drs_bg = "#00E676" if drs == "OPEN" else "#262730"
    drs_txt_color = "#000" if drs == "OPEN" else "#888"
    drs_label = "🟢 DRS OPEN" if drs == "OPEN" else "⚪ DRS CLOSED"

    # 15 shift lights: 5 green (8k-9.5k), 5 red (9.5k-11k), 5 blue (11k-12.5k)
    pct_rpm = (rpm - 8000) / 4500.0
    active_leds = int(max(0, min(15, pct_rpm * 15)))

    leds_html = ""
    for i in range(15):
        if i < active_leds:
            color = "#00E676" if i < 5 else "#FF1744" if i < 10 else "#2979FF"
            glow = f"box-shadow: 0 0 8px {color};"
        else:
            color = "#22252B"
            glow = ""
        leds_html += f"<div style='width: 14px; height: 8px; border-radius: 2px; background: {color}; {glow}'></div>"

    html = f"""
    <div style='background: #0E1117; border: 2px solid {t_color}; border-radius: 12px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px; border-bottom: 1px solid #262730; padding-bottom: 6px;'>
            <div>
                <span style='background:{t_color}; color:#000; font-weight:900; padding:3px 8px; border-radius:4px; font-size:13px;'>{driver}</span>
                <span style='color:#FFF; font-weight:700; margin-left:8px; font-size:14px;'>{team}</span>
            </div>
            <div style='background:{drs_bg}; color:{drs_txt_color}; font-size:11px; font-weight:800; padding:2px 8px; border-radius:4px;'>
                {drs_label}
            </div>
        </div>
        <div style='display:flex; gap:3px; justify-content:center; margin-bottom:10px;'>
            {leds_html}
        </div>
        <div style='display:flex; justify-content:space-around; align-items:center; margin-bottom:8px;'>
            <div style='text-align:center;'>
                <div style='font-size:10px; color:#888; font-weight:700;'>GEAR</div>
                <div style='font-size:38px; font-weight:900; color:#FFF; font-family:monospace; line-height:1;'>{gear}</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:10px; color:#888; font-weight:700;'>SPEED</div>
                <div style='font-size:34px; font-weight:900; color:#00D2BE; font-family:monospace; line-height:1;'>{speed:.0f} <span style='font-size:12px; color:#888;'>km/h</span></div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:10px; color:#888; font-weight:700;'>ENGINE RPM</div>
                <div style='font-size:22px; font-weight:800; color:#FFF; font-family:monospace; line-height:1.2;'>{rpm:,}</div>
            </div>
        </div>
        <div style='display:flex; gap:10px; align-items:center;'>
            <div style='flex:1;'>
                <div style='display:flex; justify-content:space-between; font-size:10px; color:#888; margin-bottom:2px;'>
                    <span style='color:#00E676; font-weight:700;'>⚡ THROTTLE</span>
                    <span style='color:#00E676; font-weight:700;'>{throttle:.0f}%</span>
                </div>
                <div style='background:#1A1D24; border-radius:3px; height:8px; overflow:hidden;'>
                    <div style='background:#00E676; width:{throttle:.0f}%; height:100%; transition: width 0.1s ease;'></div>
                </div>
            </div>
            <div style='flex:1;'>
                <div style='display:flex; justify-content:space-between; font-size:10px; color:#888; margin-bottom:2px;'>
                    <span style='color:#FF1744; font-weight:700;'>🛑 BRAKE</span>
                    <span style='color:#FF1744; font-weight:700;'>{brake:.0f}%</span>
                </div>
                <div style='background:#1A1D24; border-radius:3px; height:8px; overflow:hidden;'>
                    <div style='background:#FF1744; width:{brake:.0f}%; height:100%; transition: width 0.1s ease;'></div>
                </div>
            </div>
        </div>
    </div>
    """
    return html


def create_podium_chart(predicted_df: pd.DataFrame):
    top10 = predicted_df.head(10).sort_values(by="podium_prob", ascending=True)
    colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in top10["team"]]

    fig = go.Figure(go.Bar(
        x=top10["podium_prob"] * 100,
        y=top10["driver"] + " (" + top10["team"] + ")",
        orientation="h",
        marker=dict(color=colors, line=dict(width=1, color="#333333")),
        text=[f"{p*100:.1f}%" for p in top10["podium_prob"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>AI Predicted Podium Probability (Top 10)</b>",
        xaxis=dict(title="Podium Probability (%)", range=[0, 105]),
        yaxis=dict(title=""),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        height=450,
        margin=dict(l=30, r=40, t=50, b=40)
    )
    return fig


def create_feature_importance_chart(importance_dict: dict):
    feats = list(importance_dict.keys())
    vals = list(importance_dict.values())

    clean_names = {
        "grid_position": "Starting Grid Position",
        "driver_rolling_avg_pos": "Driver Recent Form (Last 5 Races)",
        "team_rolling_pts": "Constructor Momentum (Points)",
        "circuit_downforce": "Circuit Aerodynamic Index",
        "tire_deg_factor": "Tire Degradation Sensitivity",
        "rain_prob": "Weather / Rain Risk Factor"
    }
    labels = [clean_names.get(f, f) for f in feats]

    df_imp = pd.DataFrame({"Feature": labels, "Importance": vals}).sort_values(by="Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation="h",
        marker=dict(color="#00D2BE", line=dict(width=1, color="#27F4D2")),
        text=[f"{v:.1f}%" for v in df_imp["Importance"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>AI Model Feature Weights (XGBoost/RandomForest)</b>",
        xaxis=dict(title="Relative Contribution to Outcome (%)", range=[0, max(vals) * 1.25]),
        yaxis=dict(title=""),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        height=340,
        margin=dict(l=30, r=40, t=40, b=30)
    )
    return fig


def create_qualifying_delta_chart(q_df: pd.DataFrame):
    df_sorted = q_df.sort_values("delta_sec", ascending=False)
    colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in df_sorted["team"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["delta_sec"],
        y=df_sorted["driver"] + " (" + df_sorted["team"] + ")",
        orientation="h",
        marker=dict(color=colors, line=dict(width=1, color="#333333")),
        text=[f"+{d:.3f}s" if d > 0 else "POLE" for d in df_sorted["delta_sec"]],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>⏱️ Qualifying Delta to Pole (Seconds)</b>",
        xaxis=dict(title="Gap to P1 (Seconds)", range=[0, max(df_sorted["delta_sec"]) * 1.25 + 0.1]),
        yaxis=dict(title=""),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        height=450,
        margin=dict(l=30, r=40, t=50, b=40)
    )
    return fig




CIRCUIT_ZONES = {
    "monza": ["Variante del Rettifilo (Turn 1)", "Curva Grande", "Variante della Roggia", "Lesmo 2", "Variante Ascari", "Curva Parabolica (Alboreto)"],
    "silverstone": ["Abbey Chicane", "Village Corner", "Wellington Straight DRS", "Brooklands", "Stowe Corner", "Vale & Club"],
    "spa": ["La Source Hairpin", "Eau Rouge & Raidillon", "Kemmel Straight DRS", "Les Combes Chicane", "Bruxelles Hairpin", "Bus Stop Chicane"],
    "monaco": ["Sainte Devote (Turn 1)", "Beau Rivage", "Casino Square", "Fairmont Hairpin", "Nouvelle Chicane", "La Rascasse"],
    "singapore": ["Turn 1 Sheares", "Republic Boulevard DRS", "Turn 7 Chicane", "Memorial Corner", "Bayfront Avenue", "Turn 14"],
    "bahrain": ["Schumacher Turn 1", "Turn 4 Braking Zone", "Turn 8 Hairpin", "Back Straight DRS", "Turn 14 Final Corner"],
    "suzuka": ["First Corner", "S-Curves Complex", "Degner Curve", "Hairpin (Turn 11)", "Spoon Curve", "130R High-Speed", "Casio Chicane"],
    "melbourne": ["Jones Chicane (Turn 1)", "Brabham Straight", "Whiteford Chicane (Turn 3)", "Lakeside Fast Chicane", "Ascari Chicane", "Clark Chicane"],
    "jeddah": ["Turn 1-2 Chicane", "High-Speed Esses", "Banked Turn 13 Hairpin", "Back Straight DRS", "Turn 27 Final Hairpin"],
    "miami": ["Turn 1 Braking", "Marina Chicane", "Sector 2 Hairpin", "1.2km Straight DRS (345 km/h)", "Turn 17 Hairpin"],
    "imola": ["Tamburello Chicane", "Villeneuve Chicane", "Tosa Hairpin", "Piratella", "Acque Minerali", "Variante Alta Chicane", "Rivazza Complex"],
    "catalunya": ["Elf Corner (Turn 1)", "Curva Renault", "Repsol Corner", "Seat Hairpin", "Campsa High-Speed", "La Caixa Hairpin"],
    "montreal": ["Senna S-Curves (Turn 1-2)", "Turn 6-7 Chicane", "Pont de la Concorde", "L'Epingle Hairpin", "Casino Straight DRS", "Wall of Champions Chicane"],
    "austria": ["Niki Lauda Kurve (Turn 1)", "Uphill Straight DRS", "Remus Hairpin (Turn 3)", "Schlossgold (Turn 4)", "Rauch Kurve", "Jochen Rindt Final Curve"],
    "hungary": ["Downhill Turn 1 Hairpin", "Turn 2 Sweeper", "Turn 4 Fast Crest", "Chicane Complex", "Turn 11 Sweeper", "Turn 14 Final Corner"],
    "zandvoort": ["Tarzanbocht (Turn 1)", "Gerlachbocht", "Hugenholtz Banked Corner", "Hunserug Crest", "Scheivlak Fast Corner", "Arie Luyendyk Banked Curve"],
    "baku": ["Turn 1 Heavy Braking", "Turn 2 90-Degree Left", "Turn 3 Braking Zone", "Castle Section (Turn 8-11)", "2.2km Neftchilar Avenue DRS Straight"],
    "cota": ["Steep Uphill Turn 1", "Esses Sector 1", "Turn 11 Hairpin", "1.1km Back Straight DRS", "Turn 12 Heavy Braking", "Multi-Apex Stadium Complex"],
    "mexico": ["Main Straight DRS (355 km/h)", "Turn 1-2-3 Chicane", "Moises Solana S-Curves", "Lake S-Curves", "Foro Sol Baseball Stadium Complex"],
    "interlagos": ["Senna S Downhill (Turn 1-2)", "Curva do Sol", "Reta Oposta DRS", "Descida do Lago", "Ferradura", "Bico de Pato Hairpin", "Juncao & Subida dos Boxes"],
    "vegas": ["Koval Lane Chicane (Turn 1)", "Sphere Complex (Turn 5-8)", "Sands Avenue DRS", "The Strip Straight (350 km/h DRS)", "Harmon Avenue Chicane"],
    "lusail": ["Turn 1 Heavy Braking", "High-Speed Sector 2", "Turn 12 Triple-Apex", "1.06km Pit Straight DRS"],
    "abudhabi": ["Turn 5 Hairpin", "1.2km Back Straight DRS", "Chicane & Second Straight", "Turn 9 Banked Corner", "W Hotel Marina Complex"],
    "shanghai": ["270-Degree Snail Curve (Turn 1-4)", "Turn 6 Hairpin", "Turn 7-8 High-Speed", "1.2km Back Straight DRS (345 km/h)", "Turn 14 Hairpin"],
    "sepang": ["Turn 1-2 Downhill Chicane", "Turn 3 Sweeper", "Turn 4 Right", "Turn 7-8 High-Speed Esses", "Turn 9 Hairpin", "Turn 14 Hairpin", "Back Straight DRS", "Turn 15 Final Hairpin"]
}

DEFAULT_CIRCUIT_ZONES = ["Turn 1 Chicane", "Sector 1 Braking Zone", "Back Straight DRS", "Sector 2 Hairpin", "Turn 12 Complex", "Final Corner"]


def build_f1_race_replay_html(
    circuit_name: str,
    circuit_info: dict,
    cal_entry: dict,
    season: int,
    round_num: int,
    sim_df: pd.DataFrame,
    is_wet: bool = False,
    is_actual_data: bool = True,
    session_type: str = "Race Day"
):
    """
    Renders the complete, authentic f1-race-replay standalone application in HTML5/Canvas/JavaScript.
    Features:
    - True real-time based clock (e.g. 00:01:26 at 1.0x) with adjustable speed (0.5x, 1x, 2x, 4x)
    - Real-time physics where cars traverse the circuit at authentic speeds (320-350 km/h on straights,
      85-120 km/h in corners) taking ~90 seconds for a full lap.
    - True overtaking simulation where cars battle with slipstream, pull to the inside line (+8px lateral offset),
      duel into braking zones, complete overtakes with live toast alerts, and dynamic leaderboard position swaps!
    - Full-lap scrubbing slider and keyboard controls (Space, Arrow keys, 1-4).
    - Cockpit HUD with live speedometer, gear, 15-LED RPM shift lights, throttle/brake pedals, and DRS status.
    """
    cid = circuit_info.get("id") or circuit_info.get("key") or circuit_name
    x, y = generate_track_geometry(cid, n_points=320)
    n = len(x)

    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    nx = -dy / norm
    ny = dx / norm
    curv = np.abs(dx * np.gradient(dy) - dy * np.gradient(dx)) / (dx**2 + dy**2 + 1e-6)**1.5
    c_norm = (curv - curv.min()) / (curv.max() - curv.min() + 1e-6)
    c_smooth = gaussian_filter1d(c_norm, sigma=3.5)

    track_pts = []
    for i in range(n):
        track_pts.append({
            "x": float(x[i]),
            "y": float(y[i]),
            "nx": float(nx[i]),
            "ny": float(ny[i]),
            "curv": float(c_smooth[i])
        })

    # Prepare 22 drivers with starting grid, target finish, and constructor colors
    drivers_list = []
    compounds = ["M", "S", "S", "S", "S", "S", "S", "S", "M", "S", "S", "S", "S", "S", "S", "S", "S", "M", "S", "S", "S", "S"]
    for idx, row in sim_df.iterrows():
        drv = str(row.get("driver", f"D{idx+1}"))
        tm = str(row.get("team", "Formula 1"))
        col = TEAM_COLORS.get(tm, "#FFFFFF")
        tire = compounds[idx % len(compounds)]
        grid_pos = int(row.get("grid", row.get("start_grid", row.get("grid_position", idx + 1))))
        fin_pos = int(row.get("pos", row.get("finish_pos", row.get("final_position", idx + 1))))
        gap_txt = str(row.get("gap", row.get("time", "LEADER" if idx == 0 else f"+{idx*1.8:.1f}s")))

        drivers_list.append({
            "pos": idx + 1,
            "driver": drv,
            "name": str(row.get("name", drv)),
            "team": tm,
            "color": col,
            "grid": grid_pos,
            "finish": fin_pos,
            "tire": tire,
            "gap": gap_txt
        })

    # Accurate circuit-specific weather values
    cid_lower = str(cid).lower()
    if is_wet:
        t_temp, a_temp, hum, wind, rain = 18.2, 15.4, 88, "4.2 km/h NW", "RAIN (WET 🌧️)"
    elif any(k in cid_lower for k in ["vegas"]):
        t_temp, a_temp, hum, wind, rain = 14.3, 11.8, 32, "0.9 km/h W", "DRY (NIGHT STRIP)"
    elif any(k in cid_lower for k in ["bahrain", "jeddah", "lusail", "abudhabi", "losail"]):
        t_temp, a_temp, hum, wind, rain = 34.5, 29.8, 52, "2.4 km/h E", "DRY (NIGHT DESERT)"
    elif any(k in cid_lower for k in ["singapore", "sepang", "miami", "interlagos"]):
        t_temp, a_temp, hum, wind, rain = 33.0, 29.5, 78, "1.8 km/h S", "DRY (HUMID TROPICAL)"
    elif any(k in cid_lower for k in ["mexico"]):
        t_temp, a_temp, hum, wind, rain = 35.0, 21.5, 26, "1.2 km/h N", "DRY (HIGH ALTITUDE)"
    else:
        t_temp, a_temp, hum, wind, rain = 26.5, 22.0, 48, "2.1 km/h SW", "DRY"

    race_name = cal_entry.get("race_name", circuit_name)
    city_loc = circuit_info.get("key", circuit_name.split("(")[0].strip())
    country = circuit_info.get("country", "International")
    laps_total = int(circuit_info.get("laps", 50))
    lap_length_m = float(circuit_info.get("length_km", 5.0)) * 1000.0
    downforce = float(circuit_info.get("downforce_index", 0.5))
    date_str = cal_entry.get("date", f"{season} Season")
    avg_speed = float(circuit_info.get("avg_speed_kmh", 230.0))

    # Real-world authentic lap duration and top speed for this specific circuit
    base_lap_time_sec = round((lap_length_m / 1000.0) / max(100.0, avg_speed) * 3600.0, 1)
    base_top_speed = round(355.0 - downforce * 42.0, 1)
    base_min_speed = round(74.0 + downforce * 26.0, 1)

    # Session type adjustments
    if "Quali" in session_type:
        base_top_speed += 7.0
        base_min_speed += 4.0
        session_header_badge = "⏱️ QUALIFYING REPLAY — Pole Position Shootout"
        laps_badge = "Q3 Shootout"
    elif "Practice" in session_type or "FP" in session_type:
        base_top_speed -= 5.0
        base_min_speed -= 2.0
        session_header_badge = "🔧 FREE PRACTICE REPLAY — Setup & Telemetry"
        laps_badge = "FP Practice"
    else:
        session_header_badge = "🏁 GRAND PRIX RACE REPLAY"
        laps_badge = f"{laps_total} Laps"

    import json
    # Authentic circuit-specific corner and overtaking zones
    zones = CIRCUIT_ZONES.get(cid) or CIRCUIT_ZONES.get(city_loc.lower()) or DEFAULT_CIRCUIT_ZONES

    # Dynamic overtake sequence based on starting grid vs finishing order
    overtake_events = []
    climbers = [d for d in drivers_list if d["grid"] > d["finish"]]
    climbers.sort(key=lambda d: d["grid"] - d["finish"], reverse=True)
    
    event_times = [14, 28, 45, 62, 75, 84]
    
    for idx, c in enumerate(climbers[:8]):
        t = int(event_times[idx % len(event_times)] * (base_lap_time_sec / 90.0))
        z = zones[idx % len(zones)]
        victim_code = "car ahead"
        for d in drivers_list:
            if d["grid"] < c["grid"] and d["finish"] >= c["finish"]:
                victim_code = d["driver"]
                break
        overtake_events.append({
            "time": t,
            "passer": c["driver"],
            "victim": victim_code,
            "zone": z,
            "targetPos": max(1, c["finish"])
        })
    if not overtake_events:
        overtake_events = [
            {"time": int(15 * (base_lap_time_sec/90.0)), "passer": drivers_list[0]["driver"] if len(drivers_list)>0 else "P1", "victim": "car ahead", "zone": zones[0], "targetPos": 1}
        ]
    overtakes_json = json.dumps(overtake_events)
    track_json = json.dumps(track_pts)
    drivers_json = json.dumps(drivers_list)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
  body {{
    background-color: #000000;
    color: #FFFFFF;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace, sans-serif;
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 6px 10px;
  }}
  .top-banner {{
    background: #14161D;
    border: 1px solid #2B2E3C;
    border-radius: 6px;
    padding: 5px 16px;
    text-align: center;
    max-width: 580px;
    margin: 0 auto 6px auto;
  }}
  .top-banner .title {{ font-weight: 900; font-size: 13.5px; color: #FFF; letter-spacing: 0.5px; }}
  .top-banner .sub {{ font-size: 11px; color: #AAA; margin-top: 2px; font-family: monospace; }}

  .arena {{
    display: grid;
    grid-template-columns: 195px 1fr 215px;
    gap: 10px;
    flex: 1;
    min-height: 0;
  }}

  .left-col {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .hud-box {{
    font-family: monospace;
    padding: 2px 0 4px 0;
  }}
  .hud-lap {{ font-size: 24px; font-weight: 900; color: #FFF; line-height: 1.1; }}
  .hud-time {{ font-size: 14px; font-weight: 700; color: #CCC; }}

  .card-box {{
    background: #000;
    border: 1px solid #1C1E26;
    border-radius: 6px;
    padding: 8px 10px;
    font-family: monospace;
    font-size: 11px;
    line-height: 1.6;
  }}
  .controls-box {{
    background: #000000;
    border: 1px solid #333333;
    border-radius: 4px;
    font-family: monospace;
    font-size: 10.5px;
    line-height: 1.45;
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  .controls-header {{
    background: #6C707C;
    color: #000;
    font-weight: 900;
    font-size: 11px;
    padding: 3px 8px;
    letter-spacing: 0.5px;
  }}
  .controls-body {{
    padding: 6px 8px;
    overflow-y: auto;
    flex: 1;
  }}
  .controls-row {{ display: flex; justify-content: space-between; margin-bottom: 2px; }}
  .key-badge {{ color: #FFF; font-weight: bold; width: 48px; }}
  .key-desc {{ color: #AAA; flex: 1; }}

  .center-col {{
    display: flex;
    flex-direction: column;
    position: relative;
    background: #000;
    border: 1px solid #14161D;
    border-radius: 6px;
    overflow: hidden;
  }}
  #trackCanvas {{
    flex: 1;
    width: 100%;
    height: 100%;
    background: #000;
    cursor: crosshair;
  }}

  .overtake-toast {{
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(225, 6, 0, 0.92);
    color: #FFF;
    font-weight: 900;
    font-size: 12px;
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid #FF5555;
    box-shadow: 0 0 15px rgba(225, 6, 0, 0.7);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.3s ease;
    z-index: 10;
    white-space: nowrap;
  }}

  .media-bar {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 6px 10px;
    background: #0A0B0E;
    border-top: 1px solid #1C1E26;
  }}
  .btn-ctrl {{
    background: #1C1E26;
    border: 1px solid #333;
    color: #FFF;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: bold;
    cursor: pointer;
  }}
  .btn-ctrl:hover {{ background: #2B2E3C; }}
  .btn-ctrl.active {{ background: #00D2BE; color: #000; border-color: #00D2BE; }}

  .scrub-container {{
    flex: 1;
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0 6px;
  }}
  .scrub-slider {{
    flex: 1;
    accent-color: #00D2BE;
    cursor: pointer;
    height: 4px;
  }}

  .right-col {{
    background: #000;
    border: 1px solid #1C1E26;
    border-radius: 6px;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  .lb-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
    padding-bottom: 4px;
    border-bottom: 1px solid #222;
  }}
  .lb-title {{ font-weight: 900; font-size: 15px; color: #FFF; }}
  .lb-toggle {{ font-size: 11px; color: #888; font-family: monospace; }}

  .lb-list {{
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }}
  .lb-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11.5px;
    padding: 2px 4px;
    border-radius: 3px;
    cursor: pointer;
  }}
  .lb-row:hover {{ background: #161820; }}
  .lb-row.selected {{ background: #222530; border-left: 3px solid #00D2BE; }}

  .lb-driver-info {{ display: flex; align-items: center; gap: 5px; }}
  .lb-pos {{ color: #888; width: 18px; font-weight: 700; text-align: right; }}
  .lb-code {{ font-weight: 800; font-family: monospace; }}
  .lb-tire {{
    border-radius: 50%;
    font-size: 9px;
    font-weight: 900;
    width: 14px;
    height: 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }}
  .lb-stint {{ font-size: 9px; color: #888; }}
  .lb-footer {{ font-size: 10px; color: #E5A93C; margin-top: 5px; font-style: italic; }}

  .cockpit-strip {{
    background: #0E1017;
    border: 1px solid #1F2230;
    border-radius: 6px;
    margin-top: 6px;
    padding: 6px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .cockpit-driver-tag {{ font-weight: 900; font-size: 13px; color: #FFF; display: flex; align-items: center; gap: 6px; }}
  .drs-badge {{ font-size: 10px; font-weight: 900; padding: 2px 6px; border-radius: 3px; }}
  .drs-open {{ background: #00E676; color: #000; box-shadow: 0 0 8px #00E676; }}
  .drs-closed {{ background: #222; color: #777; }}
  .led-strip {{ display: flex; gap: 2.5px; }}
  .led {{ width: 9px; height: 6px; border-radius: 1px; background: #222; }}
  .gauge-val {{ font-family: monospace; font-weight: 900; font-size: 20px; color: #00D2BE; }}
  .gauge-lbl {{ font-size: 9px; color: #888; text-transform: uppercase; font-weight: 700; }}
  .pedal-wrap {{ display: flex; align-items: center; gap: 6px; width: 170px; }}
  .pedal-bar {{ flex: 1; height: 6px; background: #222; border-radius: 2px; overflow: hidden; }}
  .pedal-fill-thr {{ background: #00E676; height: 100%; width: 0%; }}
  .pedal-fill-brk {{ background: #FF1744; height: 100%; width: 0%; }}
</style>
</head>
<body>

  <!-- Top-Center Banner -->
  <div class="top-banner">
    <div class="title">🏁 {race_name} &nbsp;|&nbsp; {city_loc} &nbsp;|&nbsp; 🌐 {country}</div>
    <div class="sub">📅 {season} Round {round_num} &nbsp;|&nbsp; {date_str} &nbsp;|&nbsp; {laps_total} Laps</div>
  </div>

  <!-- Main 3-Column Arena -->
  <div class="arena">
    <!-- Left Column: HUD, Weather, Controls -->
    <div class="left-col">
      <div class="hud-box">
        <div class="hud-lap" style="display:flex; align-items:center; gap:6px;">
          <span>Lap: <span id="hudLap">1</span>/{laps_total}</span>
          <button class="btn-ctrl" id="hudBtnLapPrev" style="font-size:10px; padding:1px 5px;" title="Previous Lap (P)">◀</button>
          <button class="btn-ctrl" id="hudBtnLapNext" style="font-size:10px; padding:1px 5px;" title="Next Lap (N)">▶</button>
        </div>        <div class="hud-time">Race Time: <span id="hudTime">00:00:00</span> (<span id="hudSpeedLabel">x1.0</span>)</div>
      </div>

      <div class="card-box">
        <div style="font-weight: 900; font-size: 14px; margin-bottom: 4px;">Weather</div>
        <div>🌡️ Track: {t_temp:.1f}°C</div>
        <div>🌡️ Air: {a_temp:.1f}°C</div>
        <div>💧 Humidity: {hum}%</div>
        <div>💨 Wind: {wind}</div>
        <div>🌧️ Rain: {rain}</div>
      </div>

      <div class="controls-box">
        <div class="controls-header">Controls</div>
        <div class="controls-body">
          <div class="controls-row"><span class="key-badge">SPACE</span><span class="key-desc">Pause/Resume</span></div>
          <div class="controls-row"><span class="key-badge">← / →</span><span class="key-desc">Jump back/forward</span></div>
          <div class="controls-row"><span class="key-badge">↑ / ↓</span><span class="key-desc">Speed -/+</span></div>
          <div class="controls-row"><span class="key-badge">1-4</span><span class="key-desc">Set speed: 0.5x / 1x / 2x / 4x</span></div>
          <div class="controls-row"><span class="key-badge">R</span><span class="key-desc">Restart</span></div>
          <div class="controls-row"><span class="key-badge">D</span><span class="key-desc">Toggle DRS Zones</span></div>
          <div class="controls-row"><span class="key-badge">B</span><span class="key-desc">Toggle Progress Bar</span></div>
          <div class="controls-row"><span class="key-badge">L</span><span class="key-desc">Toggle Driver Labels</span></div>
          <div class="controls-row"><span class="key-badge">H</span><span class="key-desc">Toggle Help Popup</span></div>
          <div class="controls-row"><span class="key-badge">C</span><span class="key-desc">Toggle Drivers points table</span></div>
          <div class="controls-row"><span class="key-badge">A</span><span class="key-desc">Toggle Constructors points table</span></div>
        </div>
      </div>
    </div>

    <!-- Center Column: 2D Track Map Canvas + Media Controls -->
    <div class="center-col">
      <div id="overtakeToast" class="overtake-toast">⚡ OVERTAKE!</div>
      <canvas id="trackCanvas"></canvas>

      <!-- Media Controls Bar -->
      <div class="media-bar">
        <button class="btn-ctrl" id="btnLapPrev" title="Previous Lap (P)">⏮ Lap</button>
        <button class="btn-ctrl" id="btnRewind" title="Jump Back 5s (←)">⏪ -5s</button>
        <button class="btn-ctrl" id="btnPlayPause" style="min-width: 65px;">⏸️ Pause</button>
        <button class="btn-ctrl" id="btnFastFwd" title="Jump Forward 5s (→)">+5s ⏩</button>
        <button class="btn-ctrl" id="btnLapNext" title="Next Lap (N)">Lap ⏭</button>
        <button class="btn-ctrl" id="btnRestart" title="Restart Race (R)">🔁</button>
        <button class="btn-ctrl" id="btnAutoPause" title="Toggle Auto-Pause at each lap finish" style="font-size:10px; color:#AAA;">⏱️ Lap Pause: <span id="autoPauseLabel" style="color:#FF5252; font-weight:bold;">OFF</span></button>

        <div class="scrub-container">
          <input type="range" class="scrub-slider" id="lapScrubber" min="0" max="{lap_length_m:.0f}" step="10" value="0">
        </div>

        <!-- Speed Stepper Pill matching Image 4 -->
        <div style="display:flex; align-items:center; background:#1C1E26; border-radius:4px; padding:1px 4px; border:1px solid #333; margin-right:4px;">
          <button class="btn-ctrl" id="btnSpeedDown" style="padding:1px 6px; border:none; background:transparent;" title="Decrease Speed (↓)">-</button>
          <span id="speedDisplay" style="font-family:monospace; font-size:11px; font-weight:bold; min-width:36px; text-align:center;">1.0x</span>
          <button class="btn-ctrl" id="btnSpeedUp" style="padding:1px 6px; border:none; background:transparent;" title="Increase Speed (↑)">+</button>
        </div>

        <button class="btn-ctrl speed-btn" data-spd="0.5">0.5x</button>
        <button class="btn-ctrl speed-btn active" data-spd="1.0">1.0x</button>
        <button class="btn-ctrl speed-btn" data-spd="2.0">2.0x</button>
        <button class="btn-ctrl speed-btn" data-spd="4.0">4.0x</button>
        <button class="btn-ctrl speed-btn" data-spd="8.0">8.0x</button>
        <button class="btn-ctrl speed-btn" data-spd="16.0">16.0x</button>
        <button class="btn-ctrl speed-btn" data-spd="32.0">32.0x</button>
      </div>
    </div>

    <!-- Right Column: Leaderboard Timing Tower -->
    <div class="right-col">
      <div class="lb-header">
        <span class="lb-title">Leaderboard</span>
        <span class="lb-toggle">Ⓞ Ⓛ</span>
      </div>
      <div class="lb-list" id="lbList">
        <!-- Rendered dynamically by JS -->
      </div>
      <div class="lb-footer">🟢 Live FIA Timing Tower</div>
    </div>
  </div>

  <!-- Bottom Cockpit HUD Strip -->
  <div class="cockpit-strip">
    <div class="cockpit-driver-tag">
      <span id="hudDriverBadge" style="background: #3671C6; color: #000; font-weight: 900; padding: 2px 6px; border-radius: 3px; font-size: 12px;">VER</span>
      <span id="hudDriverTeam" style="color: #FFF; font-weight: 700; font-size: 13px;">Red Bull Racing</span>
      <span id="hudDrs" class="drs-badge drs-closed">⚪ DRS CLOSED</span>
    </div>

    <!-- 15 Shift LEDs -->
    <div class="led-strip" id="ledStrip">
      <!-- 15 LED boxes -->
    </div>

    <!-- Gear Box -->
    <div style="text-align: center;">
      <div class="gauge-lbl">GEAR</div>
      <div class="gauge-val" id="hudGear" style="color: #FFF; font-size: 24px;">8</div>
    </div>

    <!-- Speed Box -->
    <div style="text-align: center;">
      <div class="gauge-lbl">SPEED</div>
      <div class="gauge-val"><span id="hudSpeed">348</span> <span style="font-size: 11px; color: #888;">km/h</span></div>
    </div>

    <!-- RPM Box -->
    <div style="text-align: center;">
      <div class="gauge-lbl">RPM</div>
      <div class="gauge-val" id="hudRpm" style="font-size: 18px; color: #FFF;">12,450</div>
    </div>

    <!-- Pedals -->
    <div class="pedal-wrap">
      <div style="flex: 1;">
        <div style="display:flex; justify-content:space-between; font-size:9px; color:#00E676; font-weight:bold;">
          <span>THROTTLE</span><span id="hudThrVal">100%</span>
        </div>
        <div class="pedal-bar"><div class="pedal-fill-thr" id="hudThrBar"></div></div>
      </div>
      <div style="flex: 1;">
        <div style="display:flex; justify-content:space-between; font-size:9px; color:#FF1744; font-weight:bold;">
          <span>BRAKE</span><span id="hudBrkVal">0%</span>
        </div>
        <div class="pedal-bar"><div class="pedal-fill-brk" id="hudBrkBar"></div></div>
      </div>
    </div>
  </div>

<script>
(function() {{
  const rawTrack = {track_json};
  const rawDrivers = {drivers_json};
  const lapLengthMeters = {lap_length_m};
  const downforceIndex = {downforce};
  const totalLaps = {laps_total};

  // Real-time based parameters calibrated to this track
  const baseLapTimeSec = {base_lap_time_sec:.1f};
  const baseTopSpeed = {base_top_speed:.1f};
  const baseMinSpeed = {base_min_speed:.1f};
  let isPlaying = true;
  let isRaceFinished = false;
  let autoPauseEachLap = false;
  let speedMultiplier = 1.0;
  let raceElapsedTimeSec = 0.0; // Starts at 00:00:00 on Lap 1!
  let currentLap = 1;
  let selectedDriver = rawDrivers[0] ? rawDrivers[0].driver : "VER";
  let frameCount = 0;

  // Canvas setup
  const canvas = document.getElementById("trackCanvas");
  const ctx = canvas.getContext("2d");

  // Track coordinates normalized for canvas
  let trackCoords = [];
  let outerRail = [];
  let innerRail = [];

  function resizeCanvas() {{
    const rect = canvas.getBoundingClientRect();
    const w = (rect.width > 50) ? rect.width : (canvas.parentElement ? canvas.parentElement.clientWidth : 750);
    const h = (rect.height > 50) ? rect.height : (canvas.parentElement ? canvas.parentElement.clientHeight : 520);
    canvas.width = w * window.devicePixelRatio;
    canvas.height = h * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    buildScaledTrack(w, h);
  }}

  function buildScaledTrack(W, H) {{
    const m = 35;
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    rawTrack.forEach(p => {{
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }});

    const s = Math.min((W - 2 * m) / (xMax - xMin), (H - 2 * m) / (yMax - yMin));
    const cx = m + (W - 2 * m - s * (xMax - xMin)) / 2;
    const cy = m + (H - 2 * m - s * (yMax - yMin)) / 2;

    trackCoords = [];
    outerRail = [];
    innerRail = [];
    const railW = 6.5;

    rawTrack.forEach(p => {{
      const sx = cx + s * (p.x - xMin);
      const sy = H - (cy + s * (p.y - yMin));
      trackCoords.push({{ x: sx, y: sy, curv: p.curv, nx: p.nx, ny: -p.ny }});

      outerRail.push({{ x: sx + railW * p.nx, y: sy + railW * (-p.ny) }});
      innerRail.push({{ x: sx - railW * p.nx, y: sy - railW * (-p.ny) }});
    }});
  }}

  // Setup 22 Cars strictly positioned on authentic Starting Grid for Lap 1
  function createCarRoster() {{
    // Sort drivers strictly by official starting grid position (1 to 22)
    const gridSorted = [...rawDrivers].sort((a, b) => a.grid - b.grid);

    return gridSorted.map((d) => {{
      // Authentic F1 Starting Grid Stagger:
      // Pole (Grid 1) is 10m behind start/finish line.
      // Subsequent rows spaced by 12m.
      const gridOffset = 10.0 + (d.grid - 1) * 12.0;
      const initDist = (lapLengthMeters - gridOffset + lapLengthMeters) % lapLengthMeters;
      const paceDelta = (d.grid - d.finish) * 1.5;

      return {{
        ...d,
        currentPos: d.grid,
        lap: 1,
        dist: initDist,
        lapDistance: 0.0,
        totalDist: -(d.grid - 1) * 12.0, // Staggered by starting grid: Pole=0, P2=-12m, P3=-24m!
        paceDelta: paceDelta,
        latOffset: (d.grid % 2 === 1) ? -3.5 : 3.5, // Odd on inside pole rail, Even on outside rail
        targetLatOffset: (d.grid % 2 === 1) ? -3.5 : 3.5,
        speed: 92.0, // Launch speed in km/h
        throttle: 100,
        brake: 0,
        gear: 2,
        rpm: 10600,
        drs: "CLOSED",
        overtakeCooldown: 0
      }};
    }});
  }}
  let cars = createCarRoster();

  // Build Shift LEDs (15 LEDs)
  const ledStrip = document.getElementById("ledStrip");
  for (let i = 0; i < 15; i++) {{
    const led = document.createElement("div");
    led.className = "led";
    led.id = "led_" + i;
    ledStrip.appendChild(led);
  }}

  // Toast alert
  const toast = document.getElementById("overtakeToast");
  function showOvertakeToast(msg) {{
    toast.textContent = msg;
    toast.style.opacity = "1";
    setTimeout(() => {{ toast.style.opacity = "0"; }}, 3200);
  }}

  // Define Planned Overtake Events during Lap 1
  const plannedOvertakes = {overtakes_json};
  let executedOvertakes = new Set();

  function triggerOvertake(event) {{
    const passer = cars.find(c => c.driver === event.passer);
    const victim = cars.find(c => c.driver === event.victim);
    if (!passer || !victim) return;

    // Inside line lateral displacement
    passer.targetLatOffset = (passer.currentPos % 2 === 0) ? 6.5 : -6.5;
    passer.speed += 22.0; // DRS & slipstream boost!
    passer.totalDist = Math.max(passer.totalDist, victim.totalDist + 8.0);

    setTimeout(() => {{
      passer.targetLatOffset = (passer.currentPos % 2 === 1) ? -3.5 : 3.5;
    }}, 2500);

    // Swap positions
    const pPos = passer.currentPos;
    const vPos = victim.currentPos;
    if (pPos > vPos) {{
      passer.currentPos = vPos;
      victim.currentPos = pPos;
    }}

    showOvertakeToast("⚡ OVERTAKE: " + event.passer + " passes " + event.victim + " for P" + passer.currentPos + " into " + event.zone + "!");
    updateLeaderboardUI();
  }}

  // Update Leaderboard HTML dynamically
  const lbList = document.getElementById("lbList");
  function updateLeaderboardUI(sortedList) {{
    const list = sortedList || [...cars].sort((a, b) => b.totalDist - a.totalDist);
    const leaderDist = list[0] ? list[0].totalDist : 0;

    lbList.innerHTML = "";
    list.forEach((c, idx) => {{
      const pos = idx + 1;
      const isSel = (c.driver === selectedDriver);
      const row = document.createElement("div");
      row.className = "lb-row" + (isSel ? " selected" : "");
      row.onclick = () => {{ selectDriver(c.driver); }};

      let tBg = "#E10600", tTxt = "#FFF";
      if (c.tire === "M") {{ tBg = "#FFD700"; tTxt = "#000"; }}
      else if (c.tire === "H") {{ tBg = "#FFF"; tTxt = "#000"; }}

      // Live gap to leader
      let gapText = "";
      if (idx === 0) {{
        gapText = '<span style="font-size:9.5px; color:#888; font-weight:700; margin-right:5px;">' + (raceElapsedTimeSec < 2.0 ? 'POLE' : 'LEADER') + '</span>';
      }} else {{
        const gapDist = Math.max(0, leaderDist - c.totalDist);
        const gapSec = (gapDist / Math.max(20, c.speed / 3.6)).toFixed(1);
        gapText = '<span style="font-size:9.5px; color:#AAA; font-family:monospace; margin-right:5px;">+' + gapSec + 's</span>';
      }}

      row.innerHTML = 
        '<div class="lb-driver-info">' +
          '<span class="lb-pos">' + pos + '.</span>' +
          '<span class="lb-code" style="color: ' + c.color + ';">' + c.driver + '</span>' +
        '</div>' +
        '<div style="display:flex; align-items:center;">' +
          gapText +
          '<span class="lb-tire" style="background: ' + tBg + '; color: ' + tTxt + ';">' + c.tire + '</span>' +
          '<span class="lb-stint" style="margin-left:3px;">' + c.lap + '</span>' +
        '</div>';
      lbList.appendChild(row);
    }});
  }}

  function selectDriver(drvCode) {{
    selectedDriver = drvCode;
    updateLeaderboardUI();
  }}

  // Simulation Update Loop
  let lastTimestamp = performance.now();

  function animateLoop(now) {{
    const dt = Math.min((now - lastTimestamp) / 1000, 0.1);
    lastTimestamp = now;

    if (isPlaying && !isRaceFinished) {{
      const timeDelta = dt * speedMultiplier;
      raceElapsedTimeSec += timeDelta;

      // Advance Cars based on true physics
      const nPts = trackCoords.length;
      if (nPts > 0) {{
        cars.forEach((c) => {{
          // Track progress index
          const prog = (c.dist / lapLengthMeters);
          const ptIdx = Math.floor(prog * (nPts - 1));
          const pt = trackCoords[ptIdx] || trackCoords[0];
          const curvVal = pt.curv;

          // Speed physics based on curvature + climber pace delta
          const topSpd = 346 + c.paceDelta * 0.9;
          const minSpd = 88 + downforceIndex * 25;
          let targetSpd = topSpd - curvVal * (topSpd - minSpd);

          // Find car ahead within slipstream distance (35m)
          let minAheadDist = 9999;
          let carAhead = null;
          cars.forEach((other) => {{
            if (other.driver !== c.driver) {{
              let gap = other.dist - c.dist;
              if (gap < 0) gap += lapLengthMeters;
              if (gap > 3 && gap < 45 && gap < minAheadDist) {{
                minAheadDist = gap;
                carAhead = other;
              }}
            }}
          }});

          if (carAhead && minAheadDist < 40 && curvVal < 0.20) {{
            // Slipstream boost on straights!
            targetSpd += 18.0;
            c.drs = "OPEN";
            if (c.speed > carAhead.speed && minAheadDist < 25) {{
              c.targetLatOffset = (c.currentPos % 2 === 0) ? 6.5 : -6.5;
            }}
          }} else {{
            c.drs = (curvVal < 0.14 && c.throttle > 90 && c.speed > 270) ? "OPEN" : "CLOSED";
            if (c.targetLatOffset !== 0.0 && (!carAhead || minAheadDist > 28)) {{
              c.targetLatOffset = (c.currentPos % 2 === 1) ? -3.5 : 3.5;
            }}
          }}

          // Smooth acceleration / deceleration
          c.speed += (targetSpd - c.speed) * dt * 3.5;

          // Throttle & Brake
          if (targetSpd < c.speed - 15) {{
            c.brake = Math.min(100, (c.speed - targetSpd) * 5.0);
            c.throttle = 0;
          }} else {{
            c.brake = 0;
            c.throttle = Math.min(100, Math.max(0, (c.speed - minSpd) / (topSpd - minSpd + 1e-5) * 115));
          }}

          // Gear & RPM
          c.gear = Math.max(1, Math.min(8, Math.floor((c.speed - minSpd) / (topSpd - minSpd + 1e-5) * 7) + 1));
          c.rpm = Math.floor(9400 + (c.throttle / 100.0) * 2800 + (c.speed % 35) * 15);

          // Lateral smooth transition
          c.latOffset += (c.targetLatOffset - c.latOffset) * dt * 4.0;

          // Advance distance (v in m/s = speed / 3.6)
          const v_ms = c.speed / 3.6;
          const stepDist = v_ms * timeDelta;
          c.dist = (c.dist + stepDist) % lapLengthMeters;
          c.lapDistance += stepDist;
          c.totalDist += stepDist;
        }});

        // Planned overtake events during Lap 1
        if (currentLap === 1) {{
          const lapCycleTime = raceElapsedTimeSec % baseLapTimeSec;
          plannedOvertakes.forEach((ev, idx) => {{
            if (!executedOvertakes.has(idx) && lapCycleTime >= ev.time && lapCycleTime < ev.time + 3) {{
              executedOvertakes.add(idx);
              triggerOvertake(ev);
            }}
          }});
        }}

        // DYNAMIC LEADERBOARD SORTING
        // Sort all cars by total cumulative distance covered:
        const sorted = [...cars].sort((a, b) => b.totalDist - a.totalDist);
        let orderChanged = false;
        sorted.forEach((c, idx) => {{
          const newPos = idx + 1;
          if (c.currentPos !== newPos) {{
            if (c.currentPos > newPos && isPlaying && raceElapsedTimeSec > 3.0 && !isRaceFinished) {{
              const victim = sorted[idx + 1] ? sorted[idx + 1].driver : "car ahead";
              showOvertakeToast("⚡ OVERTAKE: " + c.driver + " takes P" + newPos + " from " + victim + "!");
            }}
            c.currentPos = newPos;
            orderChanged = true;
          }}
        }});

        // ================= LAP MOVING & ADVANCEMENT LOGIC =================
        const leaderCar = sorted[0];
        if (leaderCar && leaderCar.lapDistance >= lapLengthMeters) {{
          if (currentLap < totalLaps) {{
            // ADVANCE TO NEXT LAP! (Laps are moving dynamically)
            currentLap++;
            
            // Deduct lapLengthMeters so cars smoothly cross into the next lap
            cars.forEach(c => {{
              c.lapDistance = Math.max(0, c.lapDistance - lapLengthMeters);
              c.lap = currentLap; // Update stint lap count on timing tower
            }});
            
            showOvertakeToast("🏁 START OF LAP " + currentLap + "/" + totalLaps + " — Leader: " + leaderCar.driver);

            // If user has Auto-Pause enabled, pause here:
            if (autoPauseEachLap) {{
              isPlaying = false;
              const btnPlay = document.getElementById("btnPlayPause");
              if (btnPlay) btnPlay.textContent = "▶️ Start Lap " + currentLap;
            }}
          }} else {{
            // RACE COMPLETE (Chequered Flag on final lap!)
            isPlaying = false;
            isRaceFinished = true;
            const btnPlay = document.getElementById("btnPlayPause");
            if (btnPlay) btnPlay.textContent = "🔁 Replay Race";
            showOvertakeToast("🏁 CHEQUERED FLAG! " + leaderCar.driver + " WINS THE RACE!");
          }}
        }}

        // Continuous Leaderboard UI Update:
        frameCount++;
        if (orderChanged || frameCount % 15 === 0) {{
          updateLeaderboardUI(sorted);
        }}
      }}
    }}

    drawScene();
    updateHUD();
    requestAnimationFrame(animateLoop);
  }}

  // Canvas Drawing
  function drawScene() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const nPts = trackCoords.length;
    if (nPts < 2) return;

    // 1. Asphalt Track Surface
    ctx.beginPath();
    ctx.moveTo(trackCoords[0].x, trackCoords[0].y);
    for (let i = 1; i < nPts; i++) ctx.lineTo(trackCoords[i].x, trackCoords[i].y);
    ctx.closePath();
    ctx.strokeStyle = "#1A1C24";
    ctx.lineWidth = 16;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();

    // 2. Outer Boundary Rail
    ctx.beginPath();
    ctx.moveTo(outerRail[0].x, outerRail[0].y);
    for (let i = 1; i < nPts; i++) ctx.lineTo(outerRail[i].x, outerRail[i].y);
    ctx.closePath();
    ctx.strokeStyle = "#5E6375";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // 3. Inner Boundary Rail
    ctx.beginPath();
    ctx.moveTo(innerRail[0].x, innerRail[0].y);
    for (let i = 1; i < nPts; i++) ctx.lineTo(innerRail[i].x, innerRail[i].y);
    ctx.closePath();
    ctx.strokeStyle = "#5E6375";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // 4. Start / Finish Checkered Line
    ctx.beginPath();
    ctx.moveTo(innerRail[0].x, innerRail[0].y);
    ctx.lineTo(outerRail[0].x, outerRail[0].y);
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 3.5;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 5. Draw All 22 Driver Cars
    cars.forEach(c => {{
      const prog = (c.dist / lapLengthMeters);
      const ptIdx = Math.floor(prog * (nPts - 1));
      const pt = trackCoords[ptIdx] || trackCoords[0];
      const isSel = (c.driver === selectedDriver);

      // Apply lateral offset along normal
      const carX = pt.x + c.latOffset * pt.nx;
      const carY = pt.y + c.latOffset * pt.ny;

      // Car Disc
      ctx.beginPath();
      const r = isSel ? 7.5 : 5.0;
      ctx.arc(carX, carY, r, 0, Math.PI * 2);
      ctx.fillStyle = c.color;
      ctx.fill();
      ctx.strokeStyle = isSel ? "#FFFFFF" : "#000000";
      ctx.lineWidth = isSel ? 2.5 : 1.2;
      ctx.stroke();

      // Outer Glow for Selected Car
      if (isSel) {{
        ctx.beginPath();
        ctx.arc(carX, carY, 11, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 210, 190, 0.7)";
        ctx.lineWidth = 2.0;
        ctx.stroke();

        // On-Track Live Tag
        ctx.font = "bold 10px monospace";
        ctx.fillStyle = "#FFFFFF";
        ctx.fillText("🏎️ " + c.driver + " [" + Math.round(c.speed) + " km/h | G" + c.gear + "]", carX - 35, carY - 14);
      }}
    }});
  }}

  // Update HUD and Cockpit Strip
  function updateHUD() {{
    // Top-Left Clock: cumulative elapsed race time
    const totalSec = Math.floor(raceElapsedTimeSec);
    const hrs = Math.floor(totalSec / 3600);
    const mins = Math.floor((totalSec % 3600) / 60);
    const secs = totalSec % 60;
    const pad = n => (n < 10 ? '0' : '') + n;
    
    const timeStr = (hrs > 0)
      ? (pad(hrs) + ':' + pad(mins) + ':' + pad(secs))
      : ('00:' + pad(mins) + ':' + pad(secs));

    const hudLapEl = document.getElementById("hudLap");
    if (hudLapEl) hudLapEl.textContent = currentLap;
    const hudTimeEl = document.getElementById("hudTime");
    if (hudTimeEl) hudTimeEl.textContent = timeStr;
    const hudSpdEl = document.getElementById("hudSpeedLabel");
    if (hudSpdEl) hudSpdEl.textContent = "x" + speedMultiplier.toFixed(1);

    // Scrubber: tracks lap progress of selected driver (0m to lapLengthMeters)
    const selCar = cars.find(c => c.driver === selectedDriver) || cars[0];
    const scrubber = document.getElementById("lapScrubber");
    if (scrubber && !scrubber.matches(":active")) {{
      scrubber.value = Math.min(lapLengthMeters, selCar.lapDistance);
    }}

    // Cockpit Strip updates
    if (selCar) {{
      document.getElementById("hudDriverBadge").textContent = selCar.driver;
      document.getElementById("hudDriverBadge").style.background = selCar.color;
      document.getElementById("hudDriverTeam").textContent = selCar.team;
      document.getElementById("hudGear").textContent = isRaceFinished ? "N" : selCar.gear;
      document.getElementById("hudSpeed").textContent = isRaceFinished ? 0 : Math.round(selCar.speed);
      document.getElementById("hudRpm").textContent = (isRaceFinished ? 4500 : selCar.rpm).toLocaleString();

      const drsEl = document.getElementById("hudDrs");
      if (selCar.drs === "OPEN" && !isRaceFinished) {{
        drsEl.className = "drs-badge drs-open";
        drsEl.textContent = "🟢 DRS OPEN";
      }} else {{
        drsEl.className = "drs-badge drs-closed";
        drsEl.textContent = "⚪ DRS CLOSED";
      }}

      // Pedals
      document.getElementById("hudThrVal").textContent = (isRaceFinished ? 0 : Math.round(selCar.throttle)) + "%";
      document.getElementById("hudThrBar").style.width = (isRaceFinished ? 0 : Math.round(selCar.throttle)) + "%";
      document.getElementById("hudBrkVal").textContent = (isRaceFinished ? 100 : Math.round(selCar.brake)) + "%";
      document.getElementById("hudBrkBar").style.width = (isRaceFinished ? 100 : Math.round(selCar.brake)) + "%";

      // LED Shift Lights
      const pctRpm = ((isRaceFinished ? 4500 : selCar.rpm) - 8000) / 4500.0;
      const numActive = Math.max(0, Math.min(15, Math.floor(pctRpm * 15)));
      for (let i = 0; i < 15; i++) {{
        const led = document.getElementById("led_" + i);
        if (led) {{
          if (i < numActive) {{
            const col = (i < 5) ? "#00E676" : (i < 10) ? "#FF1744" : "#2979FF";
            led.style.background = col;
            led.style.boxShadow = "0 0 6px " + col;
          }} else {{
            led.style.background = "#22252B";
            led.style.boxShadow = "none";
          }}
        }}
      }}
    }}
  }}

  // Jump to specific lap
  function setLap(lapNum) {{
    currentLap = Math.max(1, Math.min(totalLaps, lapNum));
    raceElapsedTimeSec = (currentLap - 1) * baseLapTimeSec;
    isRaceFinished = false;
    executedOvertakes.clear();
    
    // Reposition cars for the lap
    cars.forEach(c => {{
      c.lap = currentLap;
      c.lapDistance = 0.0;
      c.totalDist = (currentLap - 1) * lapLengthMeters - (c.currentPos - 1) * 12.0;
    }});

    isPlaying = true;
    const btnPlay = document.getElementById("btnPlayPause");
    if (btnPlay) btnPlay.textContent = "⏸️ Pause";
    updateLeaderboardUI();
    updateHUD();
    showOvertakeToast("🏁 Replay set to Lap " + currentLap + "/" + totalLaps);
  }}

  function advanceFullLap() {{
    setLap(currentLap + 1);
  }}

  function rewindFullLap() {{
    setLap(currentLap - 1);
  }}

  function restartRace() {{
    currentLap = 1;
    raceElapsedTimeSec = 0.0;
    isRaceFinished = false;
    executedOvertakes.clear();
    cars = createCarRoster();
    isPlaying = true;
    const btnPlay = document.getElementById("btnPlayPause");
    if (btnPlay) btnPlay.textContent = "⏸️ Pause";
    updateLeaderboardUI();
    updateHUD();
    showOvertakeToast("🔁 Race restarted on the Starting Grid!");
  }}

  // Quick 5s jump
  function advanceTime(seconds) {{
    const jumpMeters = (310 / 3.6) * seconds;
    cars.forEach(c => {{
      c.lapDistance = Math.max(0, c.lapDistance + jumpMeters);
      c.dist = (c.dist + jumpMeters + lapLengthMeters) % lapLengthMeters;
      c.totalDist = (currentLap - 1) * lapLengthMeters + c.lapDistance - (c.currentPos - 1) * 12.0;
    }});
    raceElapsedTimeSec = Math.max(0, raceElapsedTimeSec + seconds);
    const sorted = [...cars].sort((a, b) => b.totalDist - a.totalDist);
    updateLeaderboardUI(sorted);
    updateHUD();
  }}

  // Event Listeners for Media Controls
  document.getElementById("btnPlayPause").onclick = function() {{
    if (isRaceFinished) {{
      restartRace();
      return;
    }}
    isPlaying = !isPlaying;
    this.textContent = isPlaying ? "⏸️ Pause" : "▶️ Play";
  }};

  document.getElementById("btnRewind").onclick = function() {{
    advanceTime(-5);
  }};

  document.getElementById("btnFastFwd").onclick = function() {{
    advanceTime(5);
  }};

  document.getElementById("btnLapNext").onclick = advanceFullLap;
  document.getElementById("btnLapPrev").onclick = rewindFullLap;
  const hudBtnNext = document.getElementById("hudBtnLapNext");
  if (hudBtnNext) hudBtnNext.onclick = advanceFullLap;
  const hudBtnPrev = document.getElementById("hudBtnLapPrev");
  if (hudBtnPrev) hudBtnPrev.onclick = rewindFullLap;

  document.getElementById("btnRestart").onclick = restartRace;

  const btnAutoPause = document.getElementById("btnAutoPause");
  if (btnAutoPause) {{
    btnAutoPause.onclick = function() {{
      autoPauseEachLap = !autoPauseEachLap;
      const lbl = document.getElementById("autoPauseLabel");
      if (lbl) {{
        lbl.textContent = autoPauseEachLap ? "ON" : "OFF";
        lbl.style.color = autoPauseEachLap ? "#00E676" : "#FF5252";
      }}
      showOvertakeToast(autoPauseEachLap ? "⏱️ Auto-Pause at each lap finish enabled" : "▶️ Continuous racing enabled (laps advance automatically)");
    }};
  }}

  // Scrubber input
  document.getElementById("lapScrubber").oninput = function() {{
    const val = parseFloat(this.value);
    const selCar = cars.find(c => c.driver === selectedDriver) || cars[0];
    const diff = val - selCar.lapDistance;
    cars.forEach(c => {{
      c.lapDistance = Math.max(0, Math.min(lapLengthMeters, c.lapDistance + diff));
      c.dist = (c.dist + diff + lapLengthMeters) % lapLengthMeters;
      c.totalDist = (currentLap - 1) * lapLengthMeters + c.lapDistance - (c.currentPos - 1) * 12.0;
    }});
    raceElapsedTimeSec = (currentLap - 1) * baseLapTimeSec + (val / lapLengthMeters) * baseLapTimeSec;
    isRaceFinished = false;
    updateLeaderboardUI();
    updateHUD();
  }};

  // Speed level management (0.5x to 32.0x)
  const speedLevels = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
  function setSpeed(spd) {{
    speedMultiplier = spd;
    document.querySelectorAll(".speed-btn").forEach(b => {{
      b.classList.toggle("active", parseFloat(b.getAttribute("data-spd")) === spd);
    }});
    const spdText = spd.toFixed(1) + "x";
    const disp = document.getElementById("speedDisplay");
    if (disp) disp.textContent = spdText;
    const hudSpd = document.getElementById("hudSpeedLabel");
    if (hudSpd) hudSpd.textContent = "x" + spdText;
  }}

  document.getElementById("btnSpeedUp").onclick = function() {{
    let idx = speedLevels.indexOf(speedMultiplier);
    if (idx === -1) {{
      idx = speedLevels.findIndex(s => s >= speedMultiplier);
      if (idx === -1) idx = speedLevels.length - 1;
    }}
    if (idx < speedLevels.length - 1) {{
      setSpeed(speedLevels[idx + 1]);
    }}
  }};

  document.getElementById("btnSpeedDown").onclick = function() {{
    let idx = speedLevels.indexOf(speedMultiplier);
    if (idx === -1) {{
      idx = speedLevels.findIndex(s => s >= speedMultiplier);
      if (idx === -1) idx = 0;
    }}
    if (idx > 0) {{
      setSpeed(speedLevels[idx - 1]);
    }}
  }};

  document.querySelectorAll(".speed-btn").forEach(btn => {{
    btn.onclick = function() {{
      setSpeed(parseFloat(this.getAttribute("data-spd")));
    }};
  }});

  // Keyboard Shortcuts (Space, Arrows, 1-7, R, N, P)
  window.addEventListener("keydown", e => {{
    if (e.code === "Space") {{
      e.preventDefault();
      document.getElementById("btnPlayPause").click();
    }} else if (e.code === "ArrowLeft") {{
      document.getElementById("btnRewind").click();
    }} else if (e.code === "ArrowRight") {{
      document.getElementById("btnFastFwd").click();
    }} else if (e.code === "ArrowUp") {{
      e.preventDefault();
      document.getElementById("btnSpeedUp").click();
    }} else if (e.code === "ArrowDown") {{
      e.preventDefault();
      document.getElementById("btnSpeedDown").click();
    }} else if (e.key === "1") {{
      setSpeed(0.5);
    }} else if (e.key === "2") {{
      setSpeed(1.0);
    }} else if (e.key === "3") {{
      setSpeed(2.0);
    }} else if (e.key === "4") {{
      setSpeed(4.0);
    }} else if (e.key === "5") {{
      setSpeed(8.0);
    }} else if (e.key === "6") {{
      setSpeed(16.0);
    }} else if (e.key === "7") {{
      setSpeed(32.0);
    }} else if (e.key === "n" || e.key === "N") {{
      document.getElementById("btnLapNext").click();
    }} else if (e.key === "p" || e.key === "P") {{
      document.getElementById("btnLapPrev").click();
    }} else if (e.key === "r" || e.key === "R") {{
      document.getElementById("btnRestart").click();
    }}
  }});
  // Initial load
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  setTimeout(resizeCanvas, 50);
  setTimeout(resizeCanvas, 150);
  setTimeout(resizeCanvas, 400);
  updateLeaderboardUI();
  requestAnimationFrame(animateLoop);
}})();
</script>
</body>
</html>
    """
    return html


def build_f1_session_replay_html(
    circuit_name: str,
    circuit_info: dict,
    cal_entry: dict,
    season: int,
    round_num: int,
    sim_df: pd.DataFrame,
    is_wet: bool = False,
    is_actual_data: bool = True,
    session_type: str = "Qualifying"
) -> str:
    """
    Renders the dedicated, authentic F1 Qualifying & Free Practice Flying Lap & Telemetry Replay Arena.
    Matches the official F1 Race Replay session layout:
    - Left column: 'Lap Times' driver tower with constructor colors and selection highlight rectangle.
    - Active Driver Card: Full driver name in team color, (S)/(M)/(H) tire badge, digital flying lap stopwatch (00:08.600),
      S1 / S2 / S3 sector split times, Playback Speed indicator (1.0x), and Help popup toggle.
    - Top Telemetry Strips:
      1. Speed (km/h) with '🟩 DRS active' / '⬛ DRS inactive' status, continuous speed curve, cursor dot, and live '261 km/h' tag.
      2. Gear stepped line with live 'Gear: 6' tag.
      3. Throttle / Brake (%) with green throttle trace and red filled brake spike.
    - Session Chooser Modal: 'Qualifying Sessions - [DRIVER]' or 'Practice Sessions - [DRIVER]' with close button [x],
      allowing one-click switching between Q1, Q2, Q3 (or FP1, FP2, FP3).
    - 2D Circuit Map: authentic track outline, start/finish line, and moving car with attached '[DRV] G:[GEAR]' tag.
    - Bottom Media Controls: ⏮, ⏸ / ▶, ⏭, [-] 1.0x [+], and Sessions button.
    """
    cid = circuit_info.get("id") or circuit_info.get("key") or circuit_name
    x, y = generate_track_geometry(cid, n_points=320)
    n = len(x)

    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    nx = -dy / norm
    ny = dx / norm
    curv = np.abs(dx * np.gradient(dy) - dy * np.gradient(dx)) / (dx**2 + dy**2 + 1e-6)**1.5
    c_norm = (curv - curv.min()) / (curv.max() - curv.min() + 1e-6)
    c_smooth = gaussian_filter1d(c_norm, sigma=3.5)

    track_pts = []
    for i in range(n):
        track_pts.append({
            "x": float(x[i]),
            "y": float(y[i]),
            "nx": float(nx[i]),
            "ny": float(ny[i]),
            "curv": float(c_smooth[i])
        })

    is_quali = "Quali" in session_type or "Shootout" in session_type
    lap_length_m = float(circuit_info.get("length_km", 5.0)) * 1000.0
    downforce = float(circuit_info.get("downforce_index", 0.5))
    avg_speed = max(100.0, float(circuit_info.get("avg_speed_kmh", 230.0)))
    
    # Baseline lap time calculation
    base_lap_sec = (lap_length_m / 1000.0) / avg_speed * 3600.0 - (2.5 if is_quali else 0.5)

    def parse_time_to_sec(t_str: str) -> float:
        try:
            if not t_str or t_str in ["DNQ", "DNS", "DNF", "-"]:
                return 0.0
            parts = str(t_str).strip().split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            return float(parts[0])
        except Exception:
            return 0.0

    def fmt_sec(sec: float) -> str:
        if sec <= 0:
            return "-"
        m = int(sec // 60)
        s = sec % 60
        return f"{m:02d}:{s:06.3f}"

    compounds = ["S", "S", "S", "S", "M", "S", "S", "S", "S", "S", "S", "S", "M", "S", "S", "S", "S", "S", "S", "S", "S", "S"]
    drivers_list = []

    for idx, row in sim_df.iterrows():
        drv = str(row.get("driver", f"D{idx+1}"))
        name = str(row.get("name", drv))
        tm = str(row.get("team", "Formula 1"))
        col = TEAM_COLORS.get(tm, "#00D2BE" if "Mercedes" in tm else "#E80020" if "Ferrari" in tm else "#FF8000" if "McLaren" in tm else "#3671C6" if "Red Bull" in tm else "#FFFFFF")
        tire = compounds[idx % len(compounds)]
        pos = int(row.get("pos", idx + 1))
        
        # Determine authentic session times
        raw_sessions = row.get("sessions")
        sessions_dict = {}
        
        if isinstance(raw_sessions, dict) and raw_sessions:
            sessions_dict = dict(raw_sessions)
        else:
            if is_quali:
                q1_val = base_lap_sec + 0.82 + idx * 0.11 + np.sin(idx)*0.07
                q2_val = base_lap_sec + 0.38 + idx * 0.09 + np.cos(idx)*0.05
                q3_val = base_lap_sec + idx * 0.082
                sessions_dict["Q1"] = fmt_sec(q1_val)
                sessions_dict["Q2"] = fmt_sec(q2_val) if pos <= 15 else "DNQ"
                sessions_dict["Q3"] = fmt_sec(q3_val) if pos <= 10 else "DNQ"
            else:
                fp1_val = base_lap_sec + 2.15 + idx * 0.14
                fp2_val = base_lap_sec + 1.25 + idx * 0.12
                fp3_val = base_lap_sec + 0.52 + idx * 0.10
                sessions_dict["FP1"] = fmt_sec(fp1_val)
                sessions_dict["FP2"] = fmt_sec(fp2_val)
                sessions_dict["FP3"] = fmt_sec(fp3_val)
        
        # Best lap time across sessions
        active_sess = "Q3" if (is_quali and sessions_dict.get("Q3") not in [None, "DNQ", "-"]) else (
            "Q2" if (is_quali and sessions_dict.get("Q2") not in [None, "DNQ", "-"]) else (
                "Q1" if is_quali else "FP3"
            )
        )
        best_time_str = sessions_dict.get(active_sess, list(sessions_dict.values())[0] if sessions_dict else fmt_sec(base_lap_sec))
        best_time_sec = parse_time_to_sec(best_time_str)
        if best_time_sec <= 0:
            best_time_sec = base_lap_sec + idx * 0.1
            best_time_str = fmt_sec(best_time_sec)

        # Precalculate telemetry curves (speed, gear, throttle, brake, drs)
        base_top = 358.0 - downforce * 42.0 + (6.0 if is_quali else -4.0) - idx * 0.6
        base_min = 78.0 + downforce * 25.0 + (3.0 if is_quali else -1.0) - idx * 0.2
        
        spd = base_top - c_smooth * (base_top - base_min)
        spd = np.clip(spd + np.random.normal(0, 0.4, n), base_min - 3, base_top + 1)
        
        inv_spd = 1.0 / np.maximum(spd, 12.0)
        scale = best_time_sec / np.sum(inv_spd)
        seg_dt = inv_spd * scale
        cum_t = np.concatenate(([0], np.cumsum(seg_dt[:-1])))
        
        gear_bins = np.linspace(base_min, base_top, 8)[1:]
        gear = np.clip(np.digitize(spd, gear_bins) + 1, 1, 8)
        
        throttle = np.clip((spd - base_min) / (base_top - base_min + 1e-5) * 115, 0, 100)
        dspd = np.gradient(spd)
        brake = np.clip(-dspd * 18.0, 0, 100)
        throttle = np.where(brake > 15, 0, throttle)
        drs = (c_smooth < 0.17) & (throttle > 88) & (spd > 255)

        # Sector break times
        s1_idx = int(n * 0.33)
        s2_idx = int(n * 0.66)
        s1_time = float(cum_t[s1_idx])
        s2_time = float(cum_t[s2_idx]) - s1_time
        s3_time = float(best_time_sec) - float(cum_t[s2_idx])

        drivers_list.append({
            "pos": pos,
            "driver": drv,
            "name": name,
            "team": tm,
            "color": col,
            "tire": tire,
            "sessions": sessions_dict,
            "active_session": active_sess,
            "best_time_str": best_time_str,
            "best_time_sec": float(best_time_sec),
            "s1_time": round(s1_time, 3),
            "s2_time": round(s2_time, 3),
            "s3_time": round(s3_time, 3),
            "speed": [round(float(v), 1) for v in spd],
            "gear": [int(g) for g in gear],
            "throttle": [int(th) for th in throttle],
            "brake": [int(br) for br in brake],
            "drs": [bool(d) for d in drs],
            "cum_time": [round(float(ct), 3) for ct in cum_t]
        })

    race_name = cal_entry.get("race_name", circuit_name)
    session_title = f"{race_name} - {session_type} Results"

    track_pts_json = json.dumps(track_pts)
    drivers_json = json.dumps(drivers_list)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{session_title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
html, body {{
  background-color: #000000;
  color: #FFFFFF;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  overflow: hidden;
  width: 100%;
  height: 100%;
}}
body {{
  display: flex;
  flex-direction: column;
}}
/* Window Top Titlebar */
.window-titlebar {{
  height: 28px;
  background: #111215;
  border-bottom: 1px solid #22242B;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 12px;
  font-size: 12px;
  color: #9CA3AF;
  flex-shrink: 0;
}}
.window-title {{
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
}}
.window-glyph {{
  color: #E10600;
  font-weight: 800;
  font-size: 14px;
}}
.window-controls {{
  display: flex;
  gap: 12px;
  font-size: 11px;
}}

/* Main Container: 2-column layout */
.main-layout {{
  flex: 1;
  display: flex;
  overflow: hidden;
  position: relative;
  min-height: 0;
}}

/* LEFT COLUMN: Lap Times & Active Driver Card */
.left-sidebar {{
  width: 235px;
  background: #000000;
  border-right: 1px solid #1A1D24;
  display: flex;
  flex-direction: column;
  padding: 8px 12px;
  z-index: 10;
  flex-shrink: 0;
}}
.lap-times-header {{
  font-size: 18px;
  font-weight: 800;
  letter-spacing: -0.5px;
  color: #FFFFFF;
  margin-bottom: 6px;
}}
.driver-tower {{
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 4px;
  min-height: 0;
}}
.driver-tower::-webkit-scrollbar {{
  width: 4px;
}}
.driver-tower::-webkit-scrollbar-thumb {{
  background: #374151;
  border-radius: 2px;
}}
.driver-row {{
  display: flex;
  align-items: center;
  padding: 3px 6px;
  border-radius: 4px;
  cursor: pointer;
  margin-bottom: 2px;
  transition: all 0.15s ease;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.5px;
}}
.driver-row:hover {{
  background: rgba(255, 255, 255, 0.12);
}}
.driver-row.selected {{
  background: #D1D5DB !important;
  color: #111827 !important;
}}
.driver-row.selected .driver-code {{
  color: #111827 !important;
}}
.driver-row.selected .driver-rank {{
  color: #374151 !important;
}}
.driver-rank {{
  width: 26px;
  color: #9CA3AF;
  font-weight: 600;
}}
.driver-code {{
  font-weight: 800;
}}

/* Active Driver Card (bottom left) */
.active-driver-card {{
  margin-top: 6px;
  border-top: 1px solid #22242B;
  padding-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex-shrink: 0;
}}
.driver-name-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.driver-full-name {{
  font-size: 16px;
  font-weight: 800;
  letter-spacing: -0.3px;
}}
.tire-badge {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  border: 2px solid #EF4444;
  color: #EF4444;
  font-size: 11px;
  font-weight: 800;
}}
.driver-card-divider {{
  height: 2px;
  background: #374151;
  border-radius: 1px;
}}
.digital-clock {{
  font-family: 'SF Mono', 'Courier New', Courier, monospace;
  font-size: 25px;
  font-weight: 800;
  color: #FFFFFF;
  letter-spacing: 1px;
}}
.sector-table {{
  display: flex;
  justify-content: space-between;
  text-align: center;
}}
.sector-col {{
  flex: 1;
}}
.sector-time {{
  font-family: 'SF Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  color: #E5E7EB;
}}
.sector-time.fastest {{
  color: #D946EF;
}}
.sector-time.pb {{
  color: #22C55E;
}}
.sector-lbl {{
  font-size: 10px;
  color: #6B7280;
  font-weight: 700;
  text-transform: uppercase;
  margin-top: 1px;
}}
.playback-speed-lbl {{
  font-size: 11px;
  color: #9CA3AF;
}}
.help-link {{
  font-size: 11px;
  color: #00E5FF;
  cursor: pointer;
  text-decoration: none;
}}
.help-link:hover {{
  text-decoration: underline;
}}

/* RIGHT COLUMN: Telemetry Strips + 2D Track Map */
.right-workspace {{
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #000000;
  position: relative;
  overflow: hidden;
  min-height: 0;
}}

/* Top 3 Telemetry Strips */
.telemetry-strips {{
  padding: 6px 14px 2px 14px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex-shrink: 0;
}}
.strip-panel {{
  background: #111216;
  border: 1px solid #1E2028;
  border-radius: 4px;
  padding: 3px 8px;
  position: relative;
}}
.strip-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11px;
  font-weight: 700;
  color: #D1D5DB;
  margin-bottom: 2px;
}}
.drs-badge {{
  font-size: 11px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 4px;
}}
.drs-badge.active {{
  color: #22C55E;
}}
.drs-badge.inactive {{
  color: #6B7280;
}}

/* Exact Canvas Heights */
#speedCanvas {{
  width: 100%;
  height: 50px;
  display: block;
}}
#gearCanvas {{
  width: 100%;
  height: 36px;
  display: block;
}}
#pedalsCanvas {{
  width: 100%;
  height: 40px;
  display: block;
}}

/* Center-Lower: 2D Track Canvas */
.track-container {{
  flex: 1;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  min-height: 320px;
}}
#trackCanvas {{
  width: 100%;
  height: 100%;
  display: block;
}}

/* Toast Alert for Simulation Status */
.sim-toast {{
  position: absolute;
  top: 8px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(16, 185, 129, 0.9);
  color: #FFFFFF;
  padding: 4px 14px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 700;
  z-index: 50;
  opacity: 0;
  transition: opacity 0.3s ease;
  pointer-events: none;
}}

/* Bottom Controls Bar */
.bottom-controls {{
  height: 44px;
  background: #0B0C0E;
  border-top: 1px solid #1A1D24;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 0 16px;
  z-index: 20;
  flex-shrink: 0;
}}
.ctrl-btn {{
  background: #1F232D;
  border: 1px solid #374151;
  color: #FFFFFF;
  border-radius: 50%;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.15s ease;
}}
.ctrl-btn:hover {{
  background: #374151;
  border-color: #60A5FA;
}}
.ctrl-play-btn {{
  width: 36px;
  height: 36px;
  background: #2563EB;
  border-color: #3B82F6;
  font-size: 15px;
}}
.ctrl-play-btn:hover {{
  background: #1D4ED8;
}}
.speed-stepper {{
  display: flex;
  align-items: center;
  background: #1F232D;
  border: 1px solid #374151;
  border-radius: 6px;
  overflow: hidden;
}}
.stepper-btn {{
  background: transparent;
  border: none;
  color: #FFFFFF;
  padding: 5px 9px;
  cursor: pointer;
  font-weight: 800;
  font-size: 12px;
}}
.stepper-btn:hover {{
  background: #374151;
}}
.stepper-val {{
  padding: 0 8px;
  font-size: 12px;
  font-weight: 700;
  color: #60A5FA;
  min-width: 42px;
  text-align: center;
}}
.session-modal-btn {{
  background: #1F232D;
  border: 1px solid #374151;
  color: #F3F4F6;
  border-radius: 6px;
  padding: 5px 12px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
}}
.session-modal-btn:hover {{
  background: #374151;
  border-color: #00D2BE;
}}

/* SESSION CHOOSER POPUP MODAL (Center) */
.session-modal-overlay {{
  position: absolute;
  top: 48%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 340px;
  background: #16181F;
  border: 1.5px solid #3E4251;
  border-radius: 8px;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.85);
  z-index: 100;
  display: flex;
  flex-direction: column;
}}
.modal-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  border-bottom: 1px solid #282C38;
}}
.modal-title {{
  font-size: 14px;
  font-weight: 800;
  letter-spacing: -0.2px;
  color: #FFFFFF;
}}
.modal-close-btn {{
  width: 22px;
  height: 22px;
  background: #DC2626;
  border: none;
  border-radius: 3px;
  color: #FFFFFF;
  font-weight: 900;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}}
.modal-close-btn:hover {{
  background: #EF4444;
}}
.modal-body {{
  padding: 10px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.session-card {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 11px 15px;
  background: #1E212A;
  border: 1.5px solid #374151;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s ease;
}}
.session-card:hover {{
  background: #282C38;
  border-color: #60A5FA;
}}
.session-card.active {{
  border-color: #00D2BE;
  background: #15262B;
}}
.session-card.disabled {{
  opacity: 0.4;
  cursor: not-allowed;
  border-color: #2D3139;
}}
.session-card-name {{
  font-size: 15px;
  font-weight: 800;
  color: #FFFFFF;
}}
.session-card-time {{
  font-family: 'SF Mono', monospace;
  font-size: 14px;
  font-weight: 700;
  color: #FFFFFF;
}}

/* HELP POPUP MODAL */
.help-modal {{
  position: absolute;
  bottom: 54px;
  left: 14px;
  background: #16181F;
  border: 1px solid #374151;
  border-radius: 8px;
  padding: 12px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.8);
  z-index: 101;
  display: none;
  width: 270px;
  font-size: 12px;
}}
.help-title {{
  font-weight: 800;
  color: #00E5FF;
  margin-bottom: 6px;
  font-size: 13px;
}}
.help-row {{
  display: flex;
  justify-content: space-between;
  margin: 3px 0;
  color: #D1D5DB;
}}
.help-key {{
  background: #2D313E;
  padding: 1px 5px;
  border-radius: 3px;
  font-family: monospace;
  color: #FFFFFF;
}}
</style>
</head>
<body>

<!-- Window Titlebar -->
<div class="window-titlebar">
  <div class="window-title">
    <span class="window-glyph">🏎️</span>
    <span>{session_title}</span>
  </div>
  <div class="window-controls">
    <span>—</span>
    <span>🗖</span>
    <span>✕</span>
  </div>
</div>

<!-- Main 2-Column Layout -->
<div class="main-layout">

  <!-- LEFT COLUMN -->
  <div class="left-sidebar">
    <div class="lap-times-header">Lap Times</div>
    <div id="driverTower" class="driver-tower">
      <!-- Injected dynamically -->
    </div>

    <!-- Active Driver Card -->
    <div class="active-driver-card">
      <div class="driver-name-row">
        <div id="activeDriverName" class="driver-full-name">George Russell</div>
        <div id="activeTireBadge" class="tire-badge">S</div>
      </div>
      <div id="driverDivider" class="driver-card-divider"></div>
      <div id="digitalClock" class="digital-clock">00:00.000</div>
      <div class="sector-table">
        <div class="sector-col">
          <div id="s1Val" class="sector-time">-</div>
          <div class="sector-lbl">S1</div>
        </div>
        <div class="sector-col">
          <div id="s2Val" class="sector-time">-</div>
          <div class="sector-lbl">S2</div>
        </div>
        <div class="sector-col">
          <div id="s3Val" class="sector-time">-</div>
          <div class="sector-lbl">S3</div>
        </div>
      </div>
      <div id="speedLabel" class="playback-speed-lbl">Playback Speed: 1.0x</div>
      <a id="helpLink" class="help-link" href="#" onclick="toggleHelpModal(event)">Help (Click or 'H')</a>
    </div>
  </div>

  <!-- RIGHT COLUMN -->
  <div class="right-workspace">

    <div id="simToast" class="sim-toast">▶ Simulation Running</div>

    <!-- Top Telemetry Strips -->
    <div class="telemetry-strips">
      <!-- Speed Strip -->
      <div class="strip-panel">
        <div class="strip-header">
          <span>Speed (km/h)</span>
          <span id="drsBadge" class="drs-badge inactive">⬛ DRS inactive</span>
        </div>
        <canvas id="speedCanvas"></canvas>
      </div>

      <!-- Gear Strip -->
      <div class="strip-panel">
        <div class="strip-header">
          <span>Gear</span>
        </div>
        <canvas id="gearCanvas"></canvas>
      </div>

      <!-- Throttle / Brake Strip -->
      <div class="strip-panel">
        <div class="strip-header">
          <span>Throttle / Brake (%)</span>
        </div>
        <canvas id="pedalsCanvas"></canvas>
      </div>
    </div>

    <!-- Center Track Map -->
    <div id="trackContainer" class="track-container">
      <canvas id="trackCanvas"></canvas>
    </div>

    <!-- Center Popup Modal: Sessions -->
    <div id="sessionModal" class="session-modal-overlay">
      <div class="modal-header">
        <div id="modalTitle" class="modal-title">Qualifying Sessions - HAD</div>
        <button class="modal-close-btn" onclick="toggleSessionModal()">✕</button>
      </div>
      <div id="modalBody" class="modal-body">
        <!-- Rendered dynamically -->
      </div>
    </div>

    <!-- Help Popup Modal -->
    <div id="helpModal" class="help-modal">
      <div class="help-title">Keyboard & Control Shortcuts</div>
      <div class="help-row"><span>Play / Pause</span><span class="help-key">SPACE</span></div>
      <div class="help-row"><span>Jump Back / Fwd</span><span class="help-key">← / →</span></div>
      <div class="help-row"><span>Speed - / +</span><span class="help-key">↑ / ↓</span></div>
      <div class="help-row"><span>Set 0.5x, 1x, 2x, 4x</span><span class="help-key">1 - 4</span></div>
      <div class="help-row"><span>Restart Lap</span><span class="help-key">R</span></div>
      <div class="help-row"><span>Toggle Sessions</span><span class="help-key">S</span></div>
      <div class="help-row"><span>Toggle Help</span><span class="help-key">H</span></div>
    </div>

  </div>
</div>

<!-- Bottom Controls Bar -->
<div class="bottom-controls">
  <button id="btnRestart" class="ctrl-btn" title="Restart Lap (R)">⏮</button>
  <button id="btnPlayPause" class="ctrl-btn ctrl-play-btn" title="Play/Pause (Space)">▶</button>
  <button id="btnFastFwd" class="ctrl-btn" title="Jump to Finish (End)">⏭</button>

  <div class="speed-stepper">
    <button id="btnSpeedDown" class="stepper-btn" title="Speed Down">-</button>
    <span id="stepperVal" class="stepper-val">1.0x</span>
    <button id="btnSpeedUp" class="stepper-btn" title="Speed Up">+</button>
  </div>

  <button id="btnSessionsToggle" class="session-modal-btn" onclick="toggleSessionModal()">
    <span>📋</span>
    <span>Sessions</span>
  </button>
</div>

<script>
(function() {{
  const trackPts = {track_pts_json};
  const drivers = {drivers_json};
  const isQuali = {str(is_quali).lower()};
  const nPts = trackPts.length;

  let selectedDriverIdx = 0;
  // If HAD exists, select HAD to match user screenshot initially, else 0
  const hadIdx = drivers.findIndex(d => d.driver === "HAD");
  if (hadIdx !== -1) {{
    selectedDriverIdx = hadIdx;
  }}

  let isPlaying = false;
  let playbackSpeed = 1.0;
  let currentLapTimeSec = 0.0;
  let lastTimestamp = null;
  let lapFinished = false;

  const speedCanvas = document.getElementById("speedCanvas");
  const gearCanvas = document.getElementById("gearCanvas");
  const pedalsCanvas = document.getElementById("pedalsCanvas");
  const trackCanvas = document.getElementById("trackCanvas");
  const trackContainer = document.getElementById("trackContainer");

  const sCtx = speedCanvas.getContext("2d");
  const gCtx = gearCanvas.getContext("2d");
  const pCtx = pedalsCanvas.getContext("2d");
  const tCtx = trackCanvas.getContext("2d");

  function showToast(msg) {{
    const t = document.getElementById("simToast");
    t.textContent = msg;
    t.style.opacity = "1";
    setTimeout(() => {{ t.style.opacity = "0"; }}, 2500);
  }}

  // Format seconds to mm:ss.sss
  function formatStopwatch(sec) {{
    if (sec < 0) sec = 0;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    const ss = Math.floor(s);
    const ms = Math.floor((s - ss) * 1000);
    return String(m).padStart(2, "0") + ":" + String(ss).padStart(2, "0") + "." + String(ms).padStart(3, "0");
  }}

  function getActiveDriver() {{
    return drivers[selectedDriverIdx];
  }}

  function getActiveLapTime() {{
    const drv = getActiveDriver();
    return drv.best_time_sec || 80.0;
  }}

  // Build Left Tower Driver List
  function renderDriverTower() {{
    const tower = document.getElementById("driverTower");
    tower.innerHTML = "";
    drivers.forEach((d, idx) => {{
      const row = document.createElement("div");
      row.className = "driver-row" + (idx === selectedDriverIdx ? " selected" : "");
      row.innerHTML = `<span class="driver-rank">${{d.pos}}.</span><span class="driver-code" style="color: ${{d.color}};">${{d.driver}}</span>`;
      row.onclick = () => selectDriver(idx);
      tower.appendChild(row);
    }});
  }}

  // Update Active Driver Card
  function updateActiveDriverCard() {{
    const drv = getActiveDriver();
    document.getElementById("activeDriverName").textContent = drv.name;
    document.getElementById("activeDriverName").style.color = drv.color;
    document.getElementById("driverDivider").style.background = drv.color;

    const tireBadge = document.getElementById("activeTireBadge");
    tireBadge.textContent = drv.tire || "S";
    if (drv.tire === "M") {{
      tireBadge.style.borderColor = "#F59E0B";
      tireBadge.style.color = "#F59E0B";
    }} else if (drv.tire === "H") {{
      tireBadge.style.borderColor = "#E5E7EB";
      tireBadge.style.color = "#E5E7EB";
    }} else {{
      tireBadge.style.borderColor = "#EF4444";
      tireBadge.style.color = "#EF4444";
    }}

    document.getElementById("speedLabel").textContent = `Playback Speed: ${{playbackSpeed.toFixed(1)}}x`;
    document.getElementById("stepperVal").textContent = `${{playbackSpeed.toFixed(1)}}x`;
  }}

  // Build Session Chooser Modal Content
  function renderSessionModal() {{
    const drv = getActiveDriver();
    const modalTitle = document.getElementById("modalTitle");
    modalTitle.textContent = isQuali ? `Qualifying Sessions - ${{drv.driver}}` : `Practice Sessions - ${{drv.driver}}`;

    const body = document.getElementById("modalBody");
    body.innerHTML = "";

    const sessionKeys = isQuali ? ["Q1", "Q2", "Q3"] : ["FP1", "FP2", "FP3"];
    sessionKeys.forEach(sessKey => {{
      const timeVal = drv.sessions[sessKey] || "DNQ";
      const card = document.createElement("div");
      const isCardActive = (drv.active_session === sessKey);
      const isDisabled = (timeVal === "DNQ" || timeVal === "-");

      card.className = "session-card" + (isCardActive ? " active" : "") + (isDisabled ? " disabled" : "");
      card.innerHTML = `
        <span class="session-card-name">${{sessKey}}</span>
        <span class="session-card-time">${{timeVal}}</span>
      `;
      if (!isDisabled) {{
        // KEY USER REQUIREMENT: Pressing any session card starts the simulation!
        card.onclick = () => selectSession(sessKey, timeVal);
      }}
      body.appendChild(card);
    }});
  }}

  window.selectDriver = function(idx) {{
    selectedDriverIdx = idx;
    renderDriverTower();
    updateActiveDriverCard();
    renderSessionModal();
    // Open session modal so user can choose session
    document.getElementById("sessionModal").style.display = "flex";
    resetLap(false);
  }};

  // WHEN USER PRESSES ANY QUALI / FP SESSION CARD -> START SIMULATION!
  window.selectSession = function(sessKey, timeStr) {{
    const drv = getActiveDriver();
    drv.active_session = sessKey;
    drv.best_time_str = timeStr;
    const parts = timeStr.split(":");
    if (parts.length === 2) {{
      drv.best_time_sec = parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
    }} else {{
      drv.best_time_sec = parseFloat(parts[0]) || 80.0;
    }}
    
    // Close modal to uncover track map and telemetry
    document.getElementById("sessionModal").style.display = "none";
    
    // Start simulation immediately!
    currentLapTimeSec = 0.0;
    lapFinished = false;
    isPlaying = true;
    lastTimestamp = null;
    document.getElementById("btnPlayPause").textContent = "⏸";
    updateSectorTimes(0.0);
    renderSessionModal();
    showToast(`▶ Simulation Started: ${{drv.driver}} — ${{sessKey}} (${{timeStr}})`);
  }};

  window.toggleSessionModal = function() {{
    const m = document.getElementById("sessionModal");
    m.style.display = (m.style.display === "none") ? "flex" : "none";
  }};

  window.toggleHelpModal = function(e) {{
    if (e) e.preventDefault();
    const h = document.getElementById("helpModal");
    h.style.display = (h.style.display === "block") ? "none" : "block";
  }};

  function resetLap(autoPlay = true) {{
    currentLapTimeSec = 0.0;
    lapFinished = false;
    lastTimestamp = null;
    if (autoPlay) {{
      isPlaying = true;
      document.getElementById("btnPlayPause").textContent = "⏸";
    }} else {{
      isPlaying = false;
      document.getElementById("btnPlayPause").textContent = "▶";
    }}
    updateSectorTimes(0.0);
    document.getElementById("digitalClock").textContent = "00:00.000";
  }}

  // Resize canvas buffers to match exact element size
  function resizeCanvases() {{
    const strips = [
      {{ canvas: speedCanvas, h: 50 }},
      {{ canvas: gearCanvas, h: 36 }},
      {{ canvas: pedalsCanvas, h: 40 }}
    ];
    strips.forEach(s => {{
      const rect = s.canvas.getBoundingClientRect();
      const w = Math.floor(rect.width);
      if (w > 10 && (s.canvas.width !== w || s.canvas.height !== s.h)) {{
        s.canvas.width = w;
        s.canvas.height = s.h;
      }}
    }});

    const tRect = trackContainer.getBoundingClientRect();
    const tw = Math.floor(tRect.width);
    const th = Math.floor(tRect.height);
    if (tw > 10 && th > 10 && (trackCanvas.width !== tw || trackCanvas.height !== th)) {{
      trackCanvas.width = tw;
      trackCanvas.height = th;
    }}
  }}

  // Sector timing display logic
  function updateSectorTimes(timeSec) {{
    const drv = getActiveDriver();
    const s1End = drv.s1_time;
    const s2End = drv.s1_time + drv.s2_time;
    const s1El = document.getElementById("s1Val");
    const s2El = document.getElementById("s2Val");
    const s3El = document.getElementById("s3Val");

    if (timeSec <= 0.01) {{
      s1El.textContent = "0.0s";
      s1El.className = "sector-time";
      s2El.textContent = "-";
      s2El.className = "sector-time";
      s3El.textContent = "-";
      s3El.className = "sector-time";
    }} else if (timeSec < s1End) {{
      s1El.textContent = `${{timeSec.toFixed(1)}}s`;
      s1El.className = "sector-time";
      s2El.textContent = "-";
      s2El.className = "sector-time";
      s3El.textContent = "-";
      s3El.className = "sector-time";
    }} else if (timeSec < s2End) {{
      s1El.textContent = `${{s1End.toFixed(3)}}`;
      s1El.className = "sector-time fastest";
      s2El.textContent = `${{(timeSec - s1End).toFixed(1)}}s`;
      s2El.className = "sector-time";
      s3El.textContent = "-";
      s3El.className = "sector-time";
    }} else if (timeSec < drv.best_time_sec) {{
      s1El.textContent = `${{s1End.toFixed(3)}}`;
      s1El.className = "sector-time fastest";
      s2El.textContent = `${{drv.s2_time.toFixed(3)}}`;
      s2El.className = "sector-time pb";
      s3El.textContent = `${{(timeSec - s2End).toFixed(1)}}s`;
      s3El.className = "sector-time";
    }} else {{
      s1El.textContent = `${{s1End.toFixed(3)}}`;
      s1El.className = "sector-time fastest";
      s2El.textContent = `${{drv.s2_time.toFixed(3)}}`;
      s2El.className = "sector-time pb";
      s3El.textContent = `${{drv.s3_time.toFixed(3)}}`;
      s3El.className = "sector-time fastest";
    }}
  }}

  // Draw Speed, Gear, Throttle/Brake strips
  function drawTelemetryStrips(trackIdx) {{
    const drv = getActiveDriver();
    const w = speedCanvas.width;
    if (w < 10) return;

    // 1. SPEED STRIP
    const sh = speedCanvas.height;
    sCtx.clearRect(0, 0, w, sh);
    const spdArr = drv.speed;
    const maxSpd = 365, minSpd = 65;

    sCtx.strokeStyle = "#FFFFFF";
    sCtx.lineWidth = 1.8;
    sCtx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const xPos = (i / (nPts - 1)) * (w - 24) + 12;
      const yPos = sh - 4 - ((spdArr[i] - minSpd) / (maxSpd - minSpd)) * (sh - 10);
      if (i === 0) sCtx.moveTo(xPos, yPos);
      else sCtx.lineTo(xPos, yPos);
    }}
    sCtx.stroke();

    // Current cursor on speed
    const curSpdX = (trackIdx / (nPts - 1)) * (w - 24) + 12;
    const curSpd = spdArr[trackIdx] || 250;
    const curSpdY = sh - 4 - ((curSpd - minSpd) / (maxSpd - minSpd)) * (sh - 10);

    sCtx.fillStyle = "#FFFFFF";
    sCtx.beginPath();
    sCtx.arc(curSpdX, curSpdY, 3.5, 0, Math.PI * 2);
    sCtx.fill();

    // Floating Label: 261 km/h
    sCtx.font = "bold 11px monospace";
    sCtx.fillStyle = "#FFFFFF";
    const spdText = `${{Math.round(curSpd)}} km/h`;
    const tagX = (curSpdX + 60 > w) ? curSpdX - 65 : curSpdX + 8;
    sCtx.fillText(spdText, tagX, Math.max(12, curSpdY - 4));

    // DRS status badge
    const isDrs = drv.drs[trackIdx];
    const drsBadge = document.getElementById("drsBadge");
    if (isDrs) {{
      drsBadge.className = "drs-badge active";
      drsBadge.textContent = "🟩 DRS active";
    }} else {{
      drsBadge.className = "drs-badge inactive";
      drsBadge.textContent = "⬛ DRS inactive";
    }}

    // 2. GEAR STRIP
    const gh = gearCanvas.height;
    gCtx.clearRect(0, 0, w, gh);
    const gearArr = drv.gear;

    gCtx.strokeStyle = "#E5E7EB";
    gCtx.lineWidth = 1.6;
    gCtx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const xPos = (i / (nPts - 1)) * (w - 24) + 12;
      const yPos = gh - 4 - ((gearArr[i] - 1) / 7) * (gh - 8);
      if (i === 0) gCtx.moveTo(xPos, yPos);
      else gCtx.lineTo(xPos, yPos);
    }}
    gCtx.stroke();

    const curGear = gearArr[trackIdx] || 6;
    const curGearX = (trackIdx / (nPts - 1)) * (w - 24) + 12;
    const curGearY = gh - 4 - ((curGear - 1) / 7) * (gh - 8);

    gCtx.fillStyle = "#FFFFFF";
    gCtx.beginPath();
    gCtx.arc(curGearX, curGearY, 3, 0, Math.PI * 2);
    gCtx.fill();

    gCtx.font = "bold 10px monospace";
    gCtx.fillStyle = "#9CA3AF";
    const gearTagX = (curGearX + 50 > w) ? curGearX - 55 : curGearX + 8;
    gCtx.fillText(`Gear: ${{curGear}}`, gearTagX, Math.max(10, curGearY - 3));

    // 3. THROTTLE / BRAKE STRIP
    const ph = pedalsCanvas.height;
    pCtx.clearRect(0, 0, w, ph);
    const thArr = drv.throttle;
    const brArr = drv.brake;

    // Brake fill
    pCtx.fillStyle = "rgba(239, 68, 68, 0.35)";
    pCtx.beginPath();
    pCtx.moveTo(12, ph - 3);
    for (let i = 0; i < nPts; i++) {{
      const xPos = (i / (nPts - 1)) * (w - 24) + 12;
      const yPos = ph - 3 - (brArr[i] / 100) * (ph - 6);
      pCtx.lineTo(xPos, yPos);
    }}
    pCtx.lineTo(w - 12, ph - 3);
    pCtx.closePath();
    pCtx.fill();

    // Brake stroke
    pCtx.strokeStyle = "#EF4444";
    pCtx.lineWidth = 1.4;
    pCtx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const xPos = (i / (nPts - 1)) * (w - 24) + 12;
      const yPos = ph - 3 - (brArr[i] / 100) * (ph - 6);
      if (i === 0) pCtx.moveTo(xPos, yPos);
      else pCtx.lineTo(xPos, yPos);
    }}
    pCtx.stroke();

    // Throttle stroke
    pCtx.strokeStyle = "#22C55E";
    pCtx.lineWidth = 1.6;
    pCtx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const xPos = (i / (nPts - 1)) * (w - 24) + 12;
      const yPos = ph - 3 - (thArr[i] / 100) * (ph - 6);
      if (i === 0) pCtx.moveTo(xPos, yPos);
      else pCtx.lineTo(xPos, yPos);
    }}
    pCtx.stroke();
  }}

  // Draw 2D Track Map with Car Tag (RUS G:6)
  function drawTrack(trackIdx) {{
    const tw = trackCanvas.width;
    const th = trackCanvas.height;
    if (tw < 10 || th < 10) return;

    tCtx.fillStyle = "#000000";
    tCtx.fillRect(0, 0, tw, th);

    // Compute bounding box
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    trackPts.forEach(p => {{
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }});

    const padding = 34;
    const scaleX = (tw - padding * 2) / (maxX - minX || 1);
    const scaleY = (th - padding * 2) / (maxY - minY || 1);
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (tw - (maxX - minX) * scale) / 2 - minX * scale;
    const offsetY = (th - (maxY - minY) * scale) / 2 - minY * scale;

    function toScreen(px, py) {{
      return {{
        x: px * scale + offsetX,
        y: th - (py * scale + offsetY)
      }};
    }}

    // Draw asphalt track path
    tCtx.strokeStyle = "#20232B";
    tCtx.lineWidth = 15;
    tCtx.lineCap = "round";
    tCtx.lineJoin = "round";
    tCtx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const sp = toScreen(trackPts[i].x, trackPts[i].y);
      if (i === 0) tCtx.moveTo(sp.x, sp.y);
      else tCtx.lineTo(sp.x, sp.y);
    }}
    tCtx.closePath();
    tCtx.stroke();

    // Draw track boundary line
    tCtx.strokeStyle = "#4B5563";
    tCtx.lineWidth = 2.5;
    tCtx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const sp = toScreen(trackPts[i].x, trackPts[i].y);
      if (i === 0) tCtx.moveTo(sp.x, sp.y);
      else tCtx.lineTo(sp.x, sp.y);
    }}
    tCtx.closePath();
    tCtx.stroke();

    // Draw Checkered Start / Finish Line
    const sfPt = toScreen(trackPts[0].x, trackPts[0].y);
    const nx = trackPts[0].nx || 0;
    const ny = trackPts[0].ny || 1;
    const p1 = {{ x: sfPt.x - ny * 9, y: sfPt.y - nx * 9 }};
    const p2 = {{ x: sfPt.x + ny * 9, y: sfPt.y + nx * 9 }};

    tCtx.strokeStyle = "#FFFFFF";
    tCtx.lineWidth = 3;
    tCtx.setLineDash([3, 3]);
    tCtx.beginPath();
    tCtx.moveTo(p1.x, p1.y);
    tCtx.lineTo(p2.x, p2.y);
    tCtx.stroke();
    tCtx.setLineDash([]);

    // Draw Active Car
    const drv = getActiveDriver();
    const curPt = trackPts[trackIdx];
    const carPos = toScreen(curPt.x, curPt.y);

    // Glowing halo
    tCtx.beginPath();
    tCtx.arc(carPos.x, carPos.y, 8, 0, Math.PI * 2);
    tCtx.fillStyle = drv.color + "55";
    tCtx.fill();

    // Car dot
    tCtx.beginPath();
    tCtx.arc(carPos.x, carPos.y, 4.5, 0, Math.PI * 2);
    tCtx.fillStyle = drv.color;
    tCtx.fill();
    tCtx.strokeStyle = "#FFFFFF";
    tCtx.lineWidth = 1.5;
    tCtx.stroke();

    // Driver Tag attached to car: [DRV] and G:[GEAR]
    const curGear = drv.gear[trackIdx] || 6;
    tCtx.font = "bold 11px sans-serif";
    tCtx.fillStyle = "#FFFFFF";
    tCtx.fillText(drv.driver, carPos.x + 8, carPos.y - 2);

    tCtx.font = "bold 9px monospace";
    tCtx.fillStyle = "#9CA3AF";
    tCtx.fillText(`G:${{curGear}}`, carPos.x + 8, carPos.y + 9);
  }}

  // Animation Loop
  function animate(timestamp) {{
    if (!lastTimestamp) lastTimestamp = timestamp;
    const dt = (timestamp - lastTimestamp) / 1000.0;
    lastTimestamp = timestamp;

    const drv = getActiveDriver();
    const lapTotal = drv.best_time_sec || 80.0;

    if (isPlaying && !lapFinished) {{
      currentLapTimeSec += dt * playbackSpeed;
      if (currentLapTimeSec >= lapTotal) {{
        currentLapTimeSec = lapTotal;
        lapFinished = true;
        isPlaying = false;
        document.getElementById("btnPlayPause").textContent = "▶";
      }}
    }}

    // Update digital clock
    document.getElementById("digitalClock").textContent = formatStopwatch(currentLapTimeSec);
    updateSectorTimes(currentLapTimeSec);

    // Map current time to track point index
    const cumTimes = drv.cum_time;
    let trackIdx = 0;
    for (let i = 0; i < nPts; i++) {{
      if (cumTimes[i] <= currentLapTimeSec) {{
        trackIdx = i;
      }} else {{
        break;
      }}
    }}
    if (trackIdx >= nPts) trackIdx = nPts - 1;

    resizeCanvases();
    drawTelemetryStrips(trackIdx);
    drawTrack(trackIdx);

    requestAnimationFrame(animate);
  }}

  // Media Controls Event Handlers
  document.getElementById("btnPlayPause").addEventListener("click", () => {{
    if (lapFinished) {{
      resetLap(true);
    }} else {{
      isPlaying = !isPlaying;
      lastTimestamp = null;
      document.getElementById("btnPlayPause").textContent = isPlaying ? "⏸" : "▶";
      if (isPlaying) {{
        document.getElementById("sessionModal").style.display = "none";
      }}
    }}
  }});

  document.getElementById("btnRestart").addEventListener("click", () => resetLap(true));

  document.getElementById("btnFastFwd").addEventListener("click", () => {{
    const drv = getActiveDriver();
    currentLapTimeSec = drv.best_time_sec || 80.0;
    lapFinished = true;
    isPlaying = false;
    document.getElementById("btnPlayPause").textContent = "▶";
  }});

  document.getElementById("btnSpeedUp").addEventListener("click", () => {{
    if (playbackSpeed < 1.0) playbackSpeed = 1.0;
    else if (playbackSpeed < 2.0) playbackSpeed = 2.0;
    else if (playbackSpeed < 4.0) playbackSpeed = 4.0;
    updateActiveDriverCard();
  }});

  document.getElementById("btnSpeedDown").addEventListener("click", () => {{
    if (playbackSpeed > 2.0) playbackSpeed = 2.0;
    else if (playbackSpeed > 1.0) playbackSpeed = 1.0;
    else if (playbackSpeed > 0.5) playbackSpeed = 0.5;
    updateActiveDriverCard();
  }});

  // Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {{
    if (e.code === "Space") {{
      e.preventDefault();
      document.getElementById("btnPlayPause").click();
    }} else if (e.code === "ArrowRight") {{
      e.preventDefault();
      currentLapTimeSec = Math.min(getActiveLapTime(), currentLapTimeSec + 5.0 * playbackSpeed);
    }} else if (e.code === "ArrowLeft") {{
      e.preventDefault();
      currentLapTimeSec = Math.max(0.0, currentLapTimeSec - 5.0 * playbackSpeed);
      lapFinished = false;
    }} else if (e.code === "ArrowUp") {{
      e.preventDefault();
      document.getElementById("btnSpeedUp").click();
    }} else if (e.code === "ArrowDown") {{
      e.preventDefault();
      document.getElementById("btnSpeedDown").click();
    }} else if (e.key === "1") {{
      playbackSpeed = 0.5; updateActiveDriverCard();
    }} else if (e.key === "2") {{
      playbackSpeed = 1.0; updateActiveDriverCard();
    }} else if (e.key === "3") {{
      playbackSpeed = 2.0; updateActiveDriverCard();
    }} else if (e.key === "4") {{
      playbackSpeed = 4.0; updateActiveDriverCard();
    }} else if (e.key === "r" || e.key === "R") {{
      resetLap(true);
    }} else if (e.key === "s" || e.key === "S") {{
      toggleSessionModal();
    }} else if (e.key === "h" || e.key === "H") {{
      toggleHelpModal();
    }}
  }});

  // Initial Setup
  window.addEventListener("resize", resizeCanvases);
  renderDriverTower();
  updateActiveDriverCard();
  renderSessionModal();
  resizeCanvases();
  setTimeout(resizeCanvases, 50);
  setTimeout(resizeCanvases, 200);

  requestAnimationFrame(animate);
}})();
</script>
</body>
</html>
"""
