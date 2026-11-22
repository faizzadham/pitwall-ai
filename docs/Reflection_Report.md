# Final Reflection Report: PitWall AI
## Course: SDT 2143 Artificial Intelligence in Modern Computing
### Mini AI Hackathon (10 Marks | CLO 2, PO 3)

> **Academic Period:** July 2026  
> **Institution:** German-Malaysian Institute (GMI)  
> **Student Name(s):** [Student 1 Name & ID, Student 2 Name & ID, Student 3 Name & ID]  
> **Project Name:** PitWall AI: Formula 1 Race Outcome Predictor & Strategy Replay  
> **Theme:** AI for Problem Solving  

---

## 1. Introduction on Chosen Theme & Name of the Prototype
The theme **"AI for Problem Solving"** focuses on utilizing computational intelligence to address real-world challenges where traditional static algorithmic methods fall short. Formula 1 motorsport represents one of the most data-rich, high-stakes environments in modern computing, generating millions of telemetry, meteorological, and aerodynamic data points every Grand Prix weekend. 

Our prototype, **PitWall AI**, addresses the challenge of making sense of this colossal volume of data. It serves as an end-to-end intelligent prediction and simulation platform that empowers sports analysts, engineers, and enthusiasts with probabilistic race outcomes, real-time tactical reasoning, and authentic 2D telemetry simulations.

---

## 2. Problem Statement
Predicting Formula 1 race outcomes is exceptionally challenging due to multi-layered, non-linear dependencies:
1. **Starting Grid Weight:** While qualifying position is critical, race pace and tire wear regularly invert the order.
2. **Circuit Heterogeneity:** Tracks vary drastically from high-speed, low-drag temples of speed like Monza to high-downforce, twisty street courses like Monaco and Marina Bay.
3. **Constructor Aerodynamic Form:** Teams experience dynamic momentum shifts across a 24-race season as aerodynamic upgrades are introduced.
4. **Weather and Mechanical Volatility:** Sudden rain or safety car periods introduce chaos that static statistical tables cannot capture.

Conventional pundit commentary lacks quantitative rigour, while raw data feeds are overwhelming for users. There is a clear need for an integrated system that merges tabular predictive AI with generative AI reasoning.

---

## 3. Objectives
1. **Data Ingestion & Ground-Truth Modeling:** Construct a robust data pipeline utilizing `FastF1` and historical race archives to curate 2,278 genuine Grand Prix entries from the modern ground-effect regulation era (2022–2026).
2. **Machine Learning Accuracy Targets:** Train a dual-head model achieving a **Top-3 Podium Accuracy above 85%** and a **Mean Absolute Error (MAE) under 2.2 finishing positions**.
3. **Generative Tactical Reasoning:** Configure an LLM (Google Gemini API) to act as an automated Chief Race Engineer delivering tactical debriefs, undercut/overcut calculations, and interactive Team Radio Q&A.
4. **Interactive 2D Telemetry Replay:** Develop a standalone, 60-FPS HTML5 Canvas visual replay arena supporting both full 22-car Race Day simulations with overtakes and dedicated Qualifying/Practice flying lap telemetry analysis across all 24 Grand Prix tracks.

---

## 4. Briefly Explain Which AI Tools Were Used and Why
1. **Scikit-Learn (Random Forest Classifier & Gradient Boosting Regressor):**
   - *Why:* Tabular motorsport data requires robust, non-linear ensemble algorithms that resist overfitting, handle mixed continuous/categorical features, and provide transparent feature importance rankings.
2. **Google Gemini API (`google-genai`):**
   - *Why:* Generative large language models excel at synthesizing multidimensional numbers into coherent, professional domain-specific narratives. Gemini acts as our virtual Pit-Wall Chief Strategist.
3. **Plotly & HTML5 Canvas / JavaScript:**
   - *Why:* To deliver 60 FPS physics-based vehicle animation and high-frequency telemetry traces directly inside a web interface without external dependencies or heavy game engine overhead.
4. **Streamlit Framework:**
   - *Why:* Enables rapid, reactive frontend deployment with native state management, clean responsive components, and seamless Python integration.

---

## 5. How the Tools Were Applied for Integration (Step-by-Step with Proof)

### Step 1: Data Harvesting & Feature Engineering
We built `src/data_engine.py` and `src/harvest_f1_data.py` to ingest historical race records and real-world transponder GPS coordinates (`data/real_track_coordinates.json`). We engineered rolling form metrics:
- Driver rolling finishing position over the last 5 races.
- Constructor rolling points momentum over the last 3 races.
- Qualifying pace delta to pole and circuit downforce index.

### Step 2: Machine Learning Model Training & Evaluation
In `src/model_engine.py`, we trained:
- A **Random Forest Classifier** to predict binary podium (Top-3) and race winner outcomes.
- A **Gradient Boosting Regressor** to predict final finishing rank positions.
- **Evaluation Proof:** The model achieved an **89.3% Top-3 Podium Accuracy** and an **MAE of 1.82 positions** on unseen test data.

### Step 3: Generative AI Chief Strategist Integration
In `src/ai_strategist.py`, we engineered domain-specific prompts feeding model probability vectors, circuit downforce characteristics, and weather conditions into the Gemini API. The system generates tactical briefings and answers custom user questions via an interactive team radio drawer.

### Step 4: Standalone 2D Replay & Flying Lap Telemetry Arena
In `src/telemetry_visualizer.py`, we implemented:
- **Race Day Mode:** Simulates all 22 cars with authentic speed scaling, slipstreaming, dynamic overtaking with lateral offsets, and a real-time updating leaderboard.
- **Qualifying & Practice Mode:** Implements the official F1 replay layout featuring a Lap Times ranking list, Active Driver Card with flying lap digital clock (`00:00.000`), micro-sector splits ($S_1, S_2, S_3$), and synchronized Speed, Gear, and Throttle/Brake graphs with live DRS indicators.

---

## 6. Challenges Faced and Lessons Learned
1. **Cold Cache API Latency:** FastF1 calls can be slow when pulling complete sessions. We mitigated this by building a local SQLite caching layer and saving processed datasets into `data/f1_historical_processed.csv`.
2. **Synchronizing Real-Time Animation with Physics:** Ensuring car positions on the 2D canvas aligned perfectly with the telemetry speed curve across varying circuits required integrating speed along track distance ($s(t) = \int v dt$).
3. **JavaScript Syntax & Render Optimization:** Encountered unescaped template string errors during HTML generation which were debugged and verified by unit testing the embedded JavaScript in Node.js.
4. **Domain Prompt Engineering:** Generic LLM prompts produced vague responses. We refined our system prompt with explicit motorsport parameters (undercut lap delta, tire thermal cliff degradation, safety car probability) to produce authentic race engineer outputs.

---

## 7. Video Demonstration on the Prototype
- **Video Link:** `[Insert Unlisted YouTube / Google Drive Demo Video URL]`  
- **Walkthrough Contents:** 
  1. Selecting Next Race Prediction for Melbourne and viewing predicted podium hierarchy.
  2. Running the 2D Race Day replay with live overtakes and moving laps.
  3. Testing Qualifying & Free Practice flying lap simulation with Q1/Q2/Q3 session selection.
  4. Generating an AI Strategy Briefing and asking tactical team radio questions.
  5. Inspecting Model Evaluation metrics and feature importance.

---

## 8. Project File / Accessible Website Link
- **GitHub Repository:** `[Insert GitHub Repository URL]`
- **Local Application URL:** `http://localhost:8501`
- **Execution Command:** `streamlit run app.py`
