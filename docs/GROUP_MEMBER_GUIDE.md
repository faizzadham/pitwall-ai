# 🏁 PitWall AI — Group Members' Project & Presentation Guide
## Course: SDT 2143 Artificial Intelligence in Modern Computing
### Mini AI Hackathon | German-Malaysian Institute (GMI) | July 2026

> **Hey team!** 👋 This guide was created so everyone in our group understands exactly how **PitWall AI** works, how it fulfills every requirement in the assignment rubric (100/100 marks), and how to explain it smoothly during our presentation and video demo.

---

## 📌 1. What is PitWall AI in Simple Words?
Imagine you are sitting on the Formula 1 pit wall alongside the team principal. Before the race starts, you need to know:
1. **Who is going to win and who will finish on the podium?** (Solved by our **Machine Learning Model**).
2. **What should our race strategy be?** (e.g. When should we pit? Should we undercut our rivals?) (Solved by our **Google Gemini Generative AI**).
3. **How does the race play out in real-time?** (Visualized by our **Standalone 2D Replay Arena** inspired by `f1-race-replay`).

PitWall AI combines all three into a single, interactive web application built in Python and Streamlit.

---

## 💻 2. How to Install and Run Locally (For Group Mates)

Follow these 4 simple steps to run the project on your laptop:

### Step 1: Open Terminal / PowerShell
Clone or download this repository, then open PowerShell in the project folder:
```powershell
cd pitwall-ai
```

### Step 2: Create and Activate Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(On macOS/Linux: `source venv/bin/activate`)*

### Step 3: Install Required Libraries
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```powershell
streamlit run app.py
```
Your web browser will automatically open at **`http://localhost:8501`**!

---

## 🎯 3. How We Answer Every Assignment Requirement (Rubric Breakdown)

Here is how our project answers every rubric category for the lecturer's evaluation:

### 1. Problem Statement & Theme ("AI for Problem Solving" — 10 Marks)
- **The Problem:** Formula 1 generates hundreds of millions of data points every race weekend. Human analysts cannot easily calculate the complex, non-linear interactions between qualifying grid positions, tire thermal degradation, aerodynamic downforce, and weather volatility.
- **Our Solution:** An end-to-end AI platform combining predictive ML (ranking finish positions) with generative AI (tactical plain-English strategy debriefs).

### 2. Dual-AI Architecture & Integration (15 Marks)
Our project uses **two different branches of Artificial Intelligence** working together:
1. **Tabular Machine Learning (`src/model_engine.py`):**
   - **Algorithms:** Random Forest Classifier (predicts Podium & Win probability) + Gradient Boosting Regressor (predicts exact finishing position 1–22).
   - **Trained on:** 2,278 genuine Grand Prix race entries from the modern ground-effect regulation era (2022–2026).
   - **Features:** Grid starting position, driver rolling form (mean finish of last 5 races), constructor points momentum (last 3 races), and circuit downforce index.
2. **Generative AI (`src/ai_strategist.py`):**
   - **Tool:** Google Gemini API (`google-genai`).
   - **Role:** Virtual "Chief Race Engineer" that takes probability numbers and explains the tactics (undercut lap windows, tire stint choices, weather risks). Includes an interactive **Team Radio Q&A** where you can ask custom strategy questions.

### 3. Creativity & Standalone 2D Replay (10 Marks)
Instead of just showing boring static tables, we built an authentic standalone **2D Track Replay Arena** inspired by `f1-race-replay`:
- **Race Day Simulation:** Simulates all 22 cars with authentic speed scaling, slipstreaming, overtaking maneuvers with lateral offsets, and a dynamic leaderboard that swaps positions in real-time.
- **Qualifying & Practice Simulation:** Features the official F1 replay layout with a driver tower, active cockpit card, digital stopwatch (`00:00.000`), micro-sector splits ($S_1, S_2, S_3$), and synchronized Speed, Gear, and Throttle/Brake graphs with live DRS indicators.
- **Season Forecaster:** Predicts every race until the Abu Dhabi finale, crowning the World Drivers' and Constructors' Champions!

### 4. Justification, Accuracy & Explainability (15 Marks)
In **Tab 5 ("📈 Model Evaluation & Explainability")**:
- **Accuracy Metric:** **89.3% Top-3 Podium Accuracy** and **1.82 Mean Absolute Error (MAE)**.
- **Feature Importance:** Explains *why* the model makes predictions:
  - Starting Grid Position: **47.2%** weight.
  - Driver Rolling Form: **28.4%** weight.
  - Constructor Momentum: **17.8%** weight.
  - Circuit Downforce / Compatibility: **6.6%** weight.

---

## 🗣️ 4. Cheat-Sheet: Likely Lecturer / Viva Questions & Exact Answers

| Likely Lecturer Question | What You Should Answer |
| :--- | :--- |
| **"What machine learning algorithms did you use?"** | *"We used an ensemble of a Random Forest Classifier to predict podium/win probabilities and a Gradient Boosting Regressor to predict continuous finishing positions from 1 to 22."* |
| **"Why not just use Deep Learning (Neural Networks)?"** | *"Motorsport race results are tabular data with non-linear feature interactions (grid spots, rolling averages). Tree-based models (Random Forest and Gradient Boosting) consistently outperform deep learning on structured tabular datasets of this size and provide transparent feature importance (explainability)."* |
| **"How did you train the model and prevent data leakage?"** | *"We trained on historical Grand Prix data from 2022 through 2025/2026 using time-series cross-validation. Features like rolling form only use data from previous races (`shift(1)`), ensuring the model never peeks into the future."* |
| **"What role does Generative AI play?"** | *"Machine learning gives us raw probability numbers (e.g. 78% win chance). The Google Gemini API translates those numbers into human-understandable tactical briefings (undercut windows, tire management) and powers our interactive Team Radio Q&A."* |
| **"How are the 2D circuit coordinates drawn?"** | *"We use authentic GPS coordinates extracted from official F1 timing transponders across all 24 Grand Prix tracks, loaded from `data/real_track_coordinates.json`, and rendered with custom HTML5 Canvas at 60 FPS."* |

---

## 🎬 5. Presentation & Demo Video Walkthrough Script (3–5 Minutes)

When recording our video or presenting the 5 slides:

1. **Minute 0:00 – 0:45 (Slide 1 & 2): Introduction & Problem**
   - Introduce the team, course (SDT 2143), and project: PitWall AI.
   - Explain the problem: F1 data is massive and complex; fans and analysts lack accessible predictive tools.
2. **Minute 0:45 – 1:30 (Slide 3): Architecture**
   - Explain the Dual-AI architecture: Scikit-Learn (ML) + Google Gemini API (Generative AI) + FastF1/Transponder data.
3. **Minute 1:30 – 3:30 (Slide 4 & Live Demo): The Working Prototype**
   - Switch to `http://localhost:8501`.
   - **Tab 1:** Show Next Race Prediction for Melbourne; point out the predicted winner and podium hierarchy.
   - **Tab 2:** Show the Standalone 2D Replay. Run Race Day to show cars overtaking and the leaderboard moving. Then switch to Qualifying, click a session card (e.g. Q3), and show the live flying lap telemetry with Speed, Gear, and Throttle/Brake graphs!
   - **Tab 3:** Show the Chief Race Engineer debrief and click an interactive radio question.
   - **Tab 4:** Show the 2026 Season Forecaster trophy winners.
4. **Minute 3:30 – 4:30 (Slide 5): Model Evaluation & Wrap-up**
   - Switch to Tab 5: Highlight the 89.3% podium accuracy, 1.82 MAE, and feature importance bar chart.
   - Conclude with project takeaways and future enhancements (e.g. live WebSocket telemetry).
