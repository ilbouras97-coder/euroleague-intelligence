@echo off
cd /d "%~dp0"
python -m src.euroleague_dashboard.ingest --start-season 2023 --end-season 2024 --datasets games,player_boxscores,team_quarter_scores,shots,player_profiles
python -m src.euroleague_dashboard.ingest --start-season 2025 --end-season 2025 --datasets games,player_boxscores,team_quarter_scores,shots,player_profiles --force-refresh
python -m src.euroleague_dashboard.features
pause
