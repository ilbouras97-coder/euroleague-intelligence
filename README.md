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

python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025

For a full refresh from the API, use this only if necessary:

python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2025 --force-refresh
