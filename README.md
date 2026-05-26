# User Instructions

## 1. Open the Project Folder

Open the project folder on your machine.

Example:

cd "path/to/Euroleague Project"

## 2. Install Dependencies Once

Open PowerShell or a terminal inside the project folder and run:

python -m pip install -r requirements.txt

## 3. Run the Web App

The easiest way on Windows is to double-click:

run_dashboard.bat

Alternatively, from PowerShell or terminal:

python -m streamlit run app.py --server.port 8501

Then open the following URL in your browser:

http://localhost:8501

## 4. Refresh Data

For a normal refresh without downloading data that already exists in the cache:

.\update_data.bat

Or:

python scripts\refresh_data.py

This refreshes historical seasons from cache, force-refreshes the current season,
rebuilds ML features, and collects the injury/availability report into:

- data/player_availability_collected.csv
- data/rotation_impact.csv

For a full refresh from the API, use this only if necessary:

python scripts\refresh_data.py --full-refresh

To refresh only the injury/availability report:

python scripts\refresh_data.py --availability-only

## 5. Automate Refreshes

On Windows, install a daily scheduled task from PowerShell:

powershell -ExecutionPolicy Bypass -File .\scripts\install_refresh_task.ps1 -Time 09:00

Scheduled-task logs are written under the local logs folder.

GitHub Actions also includes a daily `Refresh EuroLeague Data` workflow and a
manual run button. It commits refreshed generated outputs when data changes.
