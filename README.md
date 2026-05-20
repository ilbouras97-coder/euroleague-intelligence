# Euroleague Project

## Recommended Architecture

Για έναν developer, η καλύτερη αρχική επιλογή είναι **Streamlit + Python modules**.

Γιατί:

- Έχεις ένα ενιαίο Python codebase για ingestion, feature engineering, μοντέλα και UI.
- Το dashboard χτίζεται γρήγορα χωρίς ξεχωριστό React state/API layer.
- Το caching του Streamlit και το local SQLite/Parquet layer καλύπτουν άνετα ένα analytics MVP.
- Αν αργότερα χρειαστεί multi-user app, auth, background jobs ή public API, μεταφέρουμε τον ίδιο data/model layer πίσω από FastAPI και κρατάμε Streamlit ή React ως frontend.

Πρόταση φάσεων:

1. **MVP:** Streamlit, SQLite, Parquet cache, sklearn/xgboost models saved as artifacts.
2. **Production-ish:** FastAPI μόνο για inference endpoints, Streamlit/React για UI.
3. **Full product:** Postgres, scheduled ingestion, model registry, React frontend.

## Storage Design

Το Step 1 αποθηκεύει δύο επίπεδα δεδομένων:

- `data/raw/*.parquet`: immutable-ish API cache ανά dataset/season. Χρήσιμο για να μη χτυπάμε συνέχεια το API.
- `data/euroleague.sqlite`: curated tables για dashboard queries και feature engineering.

Core tables:

- `games`: πρόγραμμα, scores, home/away teams, played flag.
- `player_boxscores`: player game stats, PIR/Valuation, minutes, points, rebounds, assists.
- `team_quarter_scores`: team scores by quarter/overtime where available.
- `shots`: shot locations/events, χρήσιμο για shooting charts.
- `ingestion_runs`: metadata για κάθε ingestion run.

Οι σεζόν 2023-2026 στη Euroleague API αντιστοιχούν σε start years `2023`, `2024`, `2025`.

## Step 1 Usage

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run ingestion:

```powershell
python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025
```

Για πιο γρήγορο πρώτο test:

```powershell
python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2023 --datasets games
```

Force refresh raw cache:

```powershell
python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025 --force-refresh
```

Development ingestion για γρήγορο test:

```powershell
python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2023 --max-games-per-season 1 --force-refresh
```

## Step 2/3: Analytics Layer and Dashboard

Το project περιλαμβάνει πλέον:

- `src/euroleague_dashboard/analytics.py`: player/team game logs, season averages, overall averages, optional date filtering.
- `app.py`: Streamlit dashboard με Overview, Player Dashboard, Team Dashboard, Shot Chart και Compare.
- `assets/logos/teams`: local team logo assets extracted from the EuroLeague 2025-26 team logo pack.
- `run_dashboard.bat`: διπλό click για άνοιγμα του dashboard.
- `update_data.bat`: διπλό click για ανανέωση δεδομένων 2023-2025.

Run dashboard:

```powershell
streamlit run app.py
```

Ή σε Windows:

```powershell
.\run_dashboard.bat
```

Behavior για μέσους όρους:

- Αν δεν επιλεγεί ημερομηνία, το dashboard δείχνει συνολικό μέσο όρο για όλα τα διαθέσιμα games.
- Αν ενεργοποιηθεί `Filter by date`, οι μέσοι όροι υπολογίζονται μόνο στο επιλεγμένο date range.
- Σε κάθε περίπτωση υπάρχει ξεχωριστός πίνακας με μέσο όρο ανά σεζόν.

## Current Data Status

Το full ingestion έχει ολοκληρωθεί για seasons `2023`, `2024`, `2025`.

Τρέχοντα counts στη βάση:

- Games: 1,044
- Player boxscore rows: 29,015
- Team quarter score rows: 2,088
- Shots: 162,474

## User Instructions

### 1. Open the project folder

Ο φάκελος βρίσκεται εδώ:

```powershell
C:\Users\i.bouras\OneDrive - Systems Sunlight S.A\Desktop\Euroleague Project
```

### 2. Install dependencies once

Άνοιξε PowerShell μέσα στον φάκελο και τρέξε:

```powershell
python -m pip install -r requirements.txt
```

### 3. Run the web app

Ο πιο απλός τρόπος είναι διπλό click στο:

```powershell
run_dashboard.bat
```

Εναλλακτικά από PowerShell:

```powershell
cd "C:\Users\i.bouras\OneDrive - Systems Sunlight S.A\Desktop\Euroleague Project"
python -m streamlit run app.py --server.port 8501
```

Μετά άνοιξε στον browser:

```text
http://localhost:8501
```

### 4. Refresh data

Για κανονική ανανέωση χωρίς να ξανακατεβάζει ό,τι υπάρχει ήδη στην cache:

```powershell
.\update_data.bat
```

Ή:

```powershell
python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025
```

Για πλήρες refresh από το API, μόνο αν πραγματικά το χρειαστείς:

```powershell
python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025 --force-refresh
```

### 5. Dashboard usage

- `Overview`: improved filter pane, συνολικά KPIs, leaderboard, top players, season trend.
- `Player Dashboard`: διάλεξε παίκτη, δες μέσους όρους ανά σεζόν, game log, FG%, 3PT%, steals, blocks και turnovers.
- `Team Dashboard`: διάλεξε ομάδα, δες πλήρες όνομα/logo, leaderboard, PIR, PIR Allowed, wins/losses και game log.
- `Shot Chart`: φίλτραρε ανά ομάδα, παίκτη και made/missed shots. Made shots είναι πράσινα και missed shots κόκκινα.
- `Compare`: σύγκριση 2 ομάδων και 2 παικτών με πίνακες και spider charts.
- Sidebar `Seasons`: διάλεξε ποιες σεζόν συμμετέχουν στους υπολογισμούς.
- Sidebar `Filter by date`: αν είναι off, βλέπεις συνολικό μέσο όρο. Αν είναι on, βλέπεις μόνο το επιλεγμένο date range.
