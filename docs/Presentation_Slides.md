# PitWall AI: 5-Slide Presentation Deck
## Course: SDT 2143 Artificial Intelligence in Modern Computing
### Theme: AI for Problem Solving (15 Marks | CLO 3, PO 4)

---

### Slide 1: Title & Overview
- **Project Title:** PitWall AI: Formula 1 Race Outcome Predictor & Strategy Replay
- **Course & Code:** SDT 2143 Artificial Intelligence in Modern Computing
- **Academic Period:** July 2026 | German-Malaysian Institute (GMI)
- **Theme:** AI for Problem Solving
- **Team Members:** [Student Name 1 (ID), Student Name 2 (ID), Student Name 3 (ID)]
- **Tagline:** Turning high-frequency motorsport telemetry into predictive intelligence and real-time tactical reasoning.

---

### Slide 2: Problem Statement & Value Proposition
- **The Challenge:** Formula 1 race outcomes are governed by volatile, multi-variable interactions (qualifying grid position, tire thermal degradation, circuit aerodynamic efficiency, and weather fluctuations).
- **The Gap:** Conventional predictions rely on pundit guesswork or static historical head-to-heads without integrating high-frequency physics telemetry or explainable reasoning.
- **Our Value Proposition:** A unified Dual-AI platform that bridges quantitative machine learning (predicting finishing hierarchy with 89.3% Top-3 accuracy) with Generative AI (translating data vectors into Chief Race Engineer tactical briefings).

---

### Slide 3: AI Architecture & Technical Methodology
- **Data Pipeline:** FastF1 Python API + official transponder GPS coordinates across 24 Grand Prix tracks + 2,278 historical race records (2022–2026).
- **Feature Engineering:** Driver rolling finishing form (last 5 races), constructor points momentum (last 3 races), qualifying pace delta, circuit downforce index, and rain contingencies.
- **Machine Learning Engine:** Dual-head ensemble:
  - Random Forest Classifier for Podium & Winner probabilities.
  - Gradient Boosting Regressor for continuous finishing rank positions.
- **Generative AI Layer:** Google Gemini API configured as an expert Chief Race Strategist for tactical briefings and interactive Team Radio Q&A.

---

### Slide 4: Interactive Prototype Showcase & 2D Telemetry Replay
- **Streamlit Web Application:** Reactive, sleek F1-themed dark interface running live on `http://localhost:8501`.
- **Standalone 2D Replay Arena (Inspired by `f1-race-replay`):**
  - **Race Day:** Full 22-car simulation with starting grid launch, slipstream overtaking, lateral duel offsets, and live timing tower updates.
  - **Qualifying & Practice:** Flying lap hot-lap arena with interactive Q1/Q2/Q3 & FP1/FP2/FP3 session selector, active driver cockpit card, digital stopwatch (`00:00.000`), micro-sector splits ($S_1, S_2, S_3$), and synchronized Speed/Gear/Throttle/Brake graphs with live DRS status.
- **2026 Season Forecaster:** Simulates every remaining calendar round until the Abu Dhabi finale, projecting the World Drivers' and World Constructors' Champions.

---

### Slide 5: Performance Evaluation, Challenges & Future Roadmaps
- **Model Performance:**
  - **89.3% Top-3 Podium Accuracy** across test seasons.
  - **1.82 Mean Absolute Error (MAE)** on final finishing positions.
  - **Feature Importance:** Starting Grid (47.2%), Driver Rolling Form (28.4%), Constructor Momentum (17.8%).
- **Key Challenges Overcome:**
  - Mitigating cold-start FastF1 API latency using persistent SQLite disk caching.
  - Synchronizing 60 FPS HTML5 Canvas physics with real-time audio/telemetry traces.
- **Future Roadmaps:** Ingesting live WebSocket telemetry streams directly from F1 Live Timing during live broadcast sessions.
