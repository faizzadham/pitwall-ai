# Project Proposal: PitWall AI
## Formula 1 Race Outcome Predictor & Strategy Replay

> **Course:** SDT 2143 Artificial Intelligence in Modern Computing  
> **Institution:** German-Malaysian Institute (GMI) — Academic Period: July 2026  
> **Theme:** AI for Problem Solving  
> **Project Title:** PitWall AI: Formula 1 Next-Race Predictor & Strategy Replay  
> **Deliverable:** Project Proposal (10 Marks | CLO 2, PO 3)  

---

## 1. Project Title & Theme Justification
- **Project Title:** PitWall AI: Formula 1 Next-Race Predictor & Strategy Replay
- **Theme:** AI for Problem Solving
- **Justification:** Formula 1 represents the pinnacle of multi-variable data science and motorsport engineering. Each race weekend produces hundreds of millions of telemetry and timing data points. Predicting race outcomes and synthesizing strategy in real-time is a high-dimensional problem where traditional statistical models fail to capture non-linear interactions. PitWall AI addresses this problem by pairing tabular predictive machine learning with generative AI reasoning.

---

## 2. Problem Statement
Predicting Formula 1 race outcomes (winner, podium finishes, point scorers) is a complex challenge because outcomes depend on non-linear interactions:
1. Qualifying grid position and qualifying pace deltas.
2. Historical circuit characteristics (downforce demands, asphalt abrasiveness, power sensitivity).
3. Constructor mechanical reliability and in-season aerodynamic development momentum.
4. Tire compound degradation curves and weather volatility.
Motorsport analysts, sim-racers, and fans lack accessible, explainable tools that convert raw telemetry into probabilistic outcomes and plain-English strategic engineering insights.

---

## 3. Project Objectives
1. **Automated Telemetry Pipeline:** Ingest and process genuine historical and current-era Formula 1 session data using FastF1 and official transponder telemetry.
2. **Predictive Machine Learning Engine:** Train an ensemble model (Random Forest Classifier & Gradient Boosting Regressor) achieving >85% Top-3 podium prediction accuracy and a mean absolute error (MAE) under 2.2 finishing positions.
3. **Generative AI Race Strategist:** Integrate Google Gemini API as an automated Chief Race Engineer to translate raw probability vectors into tactical debriefs, pit stop windows, and team radio guidance.
4. **Interactive 2D Simulation Arena:** Build an interactive Streamlit web dashboard providing standalone 2D Race Day simulations (with overtakes and leaderboard updates) and dedicated Qualifying/Practice flying lap telemetry replays across 24 authentic Grand Prix circuits.

---

## 4. Scope & Target Users
- **In Scope:**
  - Modern ground-effect era data (2022 through 2026 season).
  - Multi-session coverage: Free Practice (FP), Qualifying (Q), and Race Day.
  - 24 authentic Grand Prix circuits modeled with real GPS coordinates.
  - Dual-AI architecture: Scikit-learn tabular prediction + Gemini generative reasoning.
  - Interactive full-season simulation until the Abu Dhabi finale.
- **Target Users:** Motorsport fans, sports analysts, data science students, and sim-racing strategists.

---

## 5. Technology Stack
- **Core Language:** Python 3.10+
- **Machine Learning & Analytics:** Scikit-Learn, Pandas, NumPy, Scipy
- **Motorsport Data:** FastF1 API, official transponder timing dataset
- **Generative AI:** Google Gemini API (`google-genai`)
- **Frontend & Visualization:** Streamlit, HTML5 Canvas, Plotly

---

## 6. Work Plan & Milestone Schedule
- **Milestone 1 (Days 1–3):** Problem definition, data harvesting, proposal submission.
- **Milestone 2 (Days 4–7):** Feature engineering, ML model training, cross-validation, and metric evaluation.
- **Milestone 3 (Days 8–10):** Standalone 2D replay arena, HTML5/Canvas telemetry engine, and Gemini AI integration.
- **Milestone 4 (Days 11–14):** UI polish, testing, presentation slides, reflection report, and demo video recording.
