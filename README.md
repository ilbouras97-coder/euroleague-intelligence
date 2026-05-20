# EuroLeague Dashboard Project

## Recommended Architecture

For a single developer, the best initial choice is:

Streamlit + Python modules

### Why this architecture?

- It keeps ingestion, feature engineering, modeling, and UI in a single Python codebase.
- The dashboard can be built quickly without a separate React state layer or API layer.
- Streamlit caching and a local SQLite/Parquet storage layer are more than enough for an analytics MVP.
- If the project later requires multi-user access, authentication, background jobs, or a public API, the same data/model layer can be moved behind FastAPI while keeping Streamlit or React as the frontend.

## Development Phases

### Phase 1 — MVP

- Streamlit dashboard
- SQLite curated database
- Parquet raw cache
- scikit-learn / XGBoost models saved as artifacts

### Phase 2 — Production-like Setup

- FastAPI for inference endpoints
- Streamlit or React for the UI

### Phase 3 — Full Product

- PostgreSQL database
- Scheduled ingestion jobs
- Model registry
- React frontend

---

# Storage Design

Step 1 stores data in two layers:

## Raw Data Cache

data/raw/*.parquet

This is an immutable-ish API cache per dataset and season.
It is useful because it prevents repeated calls to the EuroLeague API.

## Curated Database

data/euroleague.sqlite

This database contains curated tables used for dashboard queries and feature engineering.

## Core Tables

### games

Contains:

- Game schedule
- Scores
- Home and away teams
- Played flag

### player_boxscores

Contains player-level game statistics, including:

- PIR / Valuation
- Minutes
- Points
- Rebounds
- Assists

### team_quarter_scores

Contains team scores by quarter and overtime, where available.

### shots

Contains shot locations and shot events.
This table is useful for shooting charts.

### ingestion_runs

Contains metadata for each ingestion run.

## Season Mapping

EuroLeague API seasons from 2023 to 2026 correspond to the following start years:

2023, 2024, 2025

---

# Step 1 — Data Ingestion

## Install Dependencies

python -m pip install -r requirements.txt

## Run Full Ingestion

python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025

## Quick First Test

To run a faster test for one season and only the games dataset:

python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2023 --datasets games

## Force Refresh Raw Cache

python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025 --force-refresh

## Development Ingestion Test

Use this for a quick development test with only one game per season:

python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2023 --max-games-per-season 1 --force-refresh

---

# Steps 2/3 — Analytics Layer and Dashboard

The project now includes:

src/euroleague_dashboard/analytics.py

Provides:

- Player game logs
- Team game logs
- Season averages
- Overall averages
- Optional date filtering

app.py

Streamlit dashboard with the following sections:

- Overview
- Player Dashboard
- Team Dashboard
- Shot Chart
- Compare

assets/logos/teams

Local team logo assets extracted from the EuroLeague 2025-26 team logo pack.

run_dashboard.bat

Double-click file for opening the dashboard on Windows.

update_data.bat

Double-click file for refreshing data from seasons 2023-2025.

---

# Run the Dashboard

From the project folder:

streamlit run app.py

Or on Windows:

.\run_dashboard.bat

---

# Average Calculation Behavior

## Without Date Filter

If no date filter is selected, the dashboard shows overall averages across all available games.

## With Date Filter

If Filter by date is enabled, averages are calculated only for the selected date range.

In both cases, the dashboard also includes a separate table with averages by season.

---

# Current Data Status

Full ingestion has been completed for the following seasons:

2023, 2024, 2025

## Current Database Counts

Table | Rows
--- | ---
Games | 1,044
Player boxscore rows | 29,015
Team quarter score rows | 2,088
Shots | 162,474

---

# User Instructions

## 1. Open the Project Folder

The project folder is located here:

C:\Users\i.bouras\OneDrive - Systems Sunlight S.A\Desktop\Euroleague Project

## 2. Install Dependencies Once

Open PowerShell inside the project folder and run:

python -m pip install -r requirements.txt

## 3. Run the Web App

The easiest way is to double-click:

run_dashboard.bat

Alternatively, from PowerShell:

cd "C:\Users\i.bouras\OneDrive - Systems Sunlight S.A\Desktop\Euroleague Project"
python -m streamlit run app.py --server.port 8501

Then open the following URL in your browser:

http://localhost:8501

## 4. Refresh Data

For a normal refresh without downloading data that already exists in the cache:

.\update_data.bat

Or:

python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025

For a full refresh from the API, use this only if necessary:

python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025 --force-refresh

---

# Dashboard Usage

## Overview

Includes:

- Improved filter pane
- Overall KPIs
- Leaderboard
- Top players
- Season trend

## Player Dashboard

Select a player to view:

- Season averages
- Game log
- Field goal percentage
- Three-point percentage
- Steals
- Blocks
- Turnovers

## Team Dashboard

Select a team to view:

- Full team name
- Team logo
- Leaderboard
- PIR
- PIR allowed
- Wins and losses
- Game log

## Shot Chart

Filter by:

- Team
- Player
- Made or missed shots

Shot colors:

- Made shots are green
- Missed shots are red

## Compare

Compare:

- Two teams
- Two players

Includes comparison tables and spider charts.

## Sidebar Seasons

Select which seasons should be included in the calculations.

## Sidebar Filter by Date

- If disabled, the dashboard shows overall averages.
- If enabled, the dashboard only uses the selected date range.
