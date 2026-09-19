from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .fpl_api import FplClient


def current_gameweek(bootstrap: dict) -> int:
    event = next((item for item in bootstrap["events"] if item.get("is_current")), None)
    if event is None:
        event = next((item for item in bootstrap["events"] if item.get("is_next")), None)
    if event is None:
        raise RuntimeError("No current or next gameweek is available")
    return int(event["id"])


def collect_snapshot(team_id: int, gameweek: int | None = None) -> list[dict[str, int | float | str]]:
    client = FplClient()
    bootstrap = client.get_bootstrap()
    selected_gameweek = gameweek or current_gameweek(bootstrap)
    picks_response = client.get_picks(team_id, selected_gameweek)
    history = client.get_history(team_id)
    players = {int(player["id"]): player for player in bootstrap["elements"]}
    current_history = history.get("current", [])
    bank = float(current_history[-1].get("bank", 0)) / 10 if current_history else 0.0

    rows = []
    for pick in picks_response.get("picks", []):
        player_id = int(pick["element"])
        player = players[player_id]
        rows.append(
            {
                "season": str(bootstrap.get("season", "current")),
                "gameweek": selected_gameweek,
                "player_id": player_id,
                "starting": int(pick.get("position", 12)) <= 11,
                "selling_price": float(pick.get("selling_price", player["now_cost"])) / 10,
                "bank": bank,
            }
        )
    return rows


def append_snapshot(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["season", "gameweek", "player_id", "starting", "selling_price", "bank"]
    existing = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an FPL manager squad snapshot")
    parser.add_argument("team_id", type=int)
    parser.add_argument("--output", type=Path, default=Path("data/squad_snapshots.csv"))
    parser.add_argument("--gameweek", type=int)
    args = parser.parse_args()

    rows = collect_snapshot(args.team_id, args.gameweek)
    append_snapshot(args.output, rows)
    print(f"Recorded {len(rows)} players for team {args.team_id} in gameweek {rows[0]['gameweek']}")


if __name__ == "__main__":
    main()
