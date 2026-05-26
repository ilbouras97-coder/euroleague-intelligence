from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = "games,player_boxscores,team_quarter_scores,shots,player_profiles"


def run_step(name: str, args: list[str]) -> None:
    started = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{started}] START {name}", flush=True)
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)
    finished = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{finished}] DONE  {name}", flush=True)


def ingest_args(
    start_season: int,
    end_season: int,
    datasets: str,
    force_refresh: bool,
) -> list[str]:
    args = [
        "-m",
        "src.euroleague_dashboard.ingest",
        "--start-season",
        str(start_season),
        "--end-season",
        str(end_season),
        "--datasets",
        datasets,
    ]
    if force_refresh:
        args.append("--force-refresh")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh EuroLeague data, ML features and player availability outputs."
    )
    parser.add_argument("--start-season", type=int, default=2023)
    parser.add_argument("--current-season", type=int, default=2025)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Force refresh every selected season instead of only the current season.",
    )
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-availability", action="store_true")
    parser.add_argument(
        "--availability-only",
        action="store_true",
        help="Collect injury/availability data without refreshing boxscores or features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.current_season < args.start_season:
        raise SystemExit("--current-season must be greater than or equal to --start-season")

    skip_ingest = args.skip_ingest or args.availability_only
    skip_features = args.skip_features or args.availability_only

    if not skip_ingest:
        if args.full_refresh:
            run_step(
                "ingest all seasons",
                ingest_args(args.start_season, args.current_season, args.datasets, True),
            )
        else:
            historical_end = args.current_season - 1
            if historical_end >= args.start_season:
                run_step(
                    "ingest historical seasons from cache",
                    ingest_args(args.start_season, historical_end, args.datasets, False),
                )
            run_step(
                "ingest current season from API",
                ingest_args(args.current_season, args.current_season, args.datasets, True),
            )

    if not skip_features:
        run_step("build ML features", ["-m", "src.euroleague_dashboard.features"])

    if not args.skip_availability:
        run_step("collect player availability", ["-m", "src.euroleague_dashboard.availability_collector"])

    print("Refresh complete.", flush=True)


if __name__ == "__main__":
    main()
