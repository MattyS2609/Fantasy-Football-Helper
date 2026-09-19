from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .model import FEATURE_NAMES, TARGET_NAME

GAMEWEEK_PATTERN = re.compile(r"gw(\d+)\.csv$", re.IGNORECASE)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_gameweeks(season_path: Path) -> pd.DataFrame:
    files = []
    for path in (season_path / "gws").glob("gw*.csv"):
        match = GAMEWEEK_PATTERN.search(path.name)
        if match:
            files.append((int(match.group(1)), path))
    if not files:
        raise FileNotFoundError(f"No gameweek CSV files found in {season_path / 'gws'}")

    frames = []
    for gameweek, path in sorted(files):
        frame = pd.read_csv(path)
        frame["gameweek"] = gameweek
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["element"] = pd.to_numeric(result["element"], errors="coerce")
    return result.dropna(subset=["element"]).copy()


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _season_profile(history: pd.DataFrame) -> dict[str, dict[str, float]]:
    if "name" not in history:
        return {}
    profiles = {}
    for name, player_rows in history.groupby(history["name"].astype(str).str.lower().str.strip()):
        minutes = _series(player_rows, "minutes").sum()
        starts = _series(player_rows, "starts").sum()
        profiles[name] = {
            "previous_season_points": _series(player_rows, "total_points").sum(),
            "previous_season_minutes": minutes,
            "previous_season_starts": starts,
            "previous_season_points_per_90": _series(player_rows, "total_points").sum() / max(1.0, minutes / 90),
            "previous_season_expected_goals": _series(player_rows, "expected_goals").sum(),
            "previous_season_expected_assists": _series(player_rows, "expected_assists").sum(),
            "previous_season_clean_sheets": _series(player_rows, "clean_sheets").sum(),
        }
    return profiles


@lru_cache(maxsize=None)
def _load_fixtures(season_path: Path) -> pd.DataFrame | None:
    fixtures_path = season_path / "fixtures.csv"
    if not fixtures_path.exists():
        return None
    return pd.read_csv(fixtures_path)


@lru_cache(maxsize=None)
def _fixture_maps(season_path: Path) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], bool]]:
    fixtures = _load_fixtures(season_path)
    difficulty_map: dict[tuple[int, int], float] = {}
    home_map: dict[tuple[int, int], bool] = {}
    if fixtures is None:
        return difficulty_map, home_map
    for _, fixture in fixtures.iterrows():
        if pd.isna(fixture.get("event")):
            continue
        event = int(fixture["event"])
        home_team = int(fixture["team_h"])
        away_team = int(fixture["team_a"])
        difficulty_map[(event, home_team)] = _number(fixture.get("team_h_difficulty"), 3.0)
        difficulty_map[(event, away_team)] = _number(fixture.get("team_a_difficulty"), 3.0)
        home_map[(event, home_team)] = True
        home_map[(event, away_team)] = False
    return difficulty_map, home_map


@lru_cache(maxsize=None)
def _load_teams(season_path: Path) -> pd.DataFrame | None:
    teams_path = season_path / "teams.csv"
    return pd.read_csv(teams_path) if teams_path.exists() else None


@lru_cache(maxsize=None)
def _team_maps(season_path: Path) -> tuple[dict[str, int], dict[str, float]]:
    teams = _load_teams(season_path)
    if teams is None:
        return {}, {}
    return (
        dict(zip(teams["name"].astype(str), teams["id"].astype(int))),
        dict(zip(teams["name"].astype(str), teams["strength"].astype(float))),
    )


def _team_id(season_path: Path, team_name: Any) -> int:
    mapping, _ = _team_maps(season_path)
    return int(mapping.get(str(team_name), 0))


def _team_strength(season_path: Path, team_name: Any) -> float:
    _, mapping = _team_maps(season_path)
    return _number(mapping.get(str(team_name)), 3.0)


def _fixture_difficulty(season_path: Path, gameweek: int, team: int) -> float:
    difficulty_map, _ = _fixture_maps(season_path)
    return difficulty_map.get((gameweek, team), 3.0)


@lru_cache(maxsize=None)
def _fixture_context(season_path: Path, first_gameweek: int, team: int) -> tuple[float, float]:
    _, home_map = _fixture_maps(season_path)
    if not team:
        return 0.0, 0.0
    matches = [(week, is_home) for (week, fixture_team), is_home in home_map.items()
               if fixture_team == team and first_gameweek <= week < first_gameweek + 5]
    if not matches:
        return 0.0, 0.0
    return len(matches) / 5, sum(is_home for _, is_home in matches) / len(matches)


def _rolling_features(history: pd.DataFrame, season_path: Path, gameweek: int, player_id: int) -> dict[str, float]:
    previous = history[(history["element"] == player_id) & (history["gameweek"] < gameweek)].sort_values("gameweek")
    recent = previous.tail(5)
    recent_three = previous.tail(3)
    if recent.empty:
        return {}

    minutes = _series(recent, "minutes").sum()
    starts = ( _series(recent, "minutes") >= 60).sum()
    team_name = previous.iloc[-1].get("team")
    team = _team_id(season_path, team_name)
    position = int(_number(previous.iloc[-1].get("element_type", previous.iloc[-1].get("position"))))
    price = _number(previous.iloc[-1].get("value", previous.iloc[-1].get("now_cost")))
    if price > 20:
        price /= 10

    expected_minutes = 45.0 if minutes <= 0 or starts <= 0 else min(90.0, minutes / starts)
    rotation_probability = 1.0 - expected_minutes / 90.0
    fixture_congestion, home_fixture_ratio = _fixture_context(season_path, gameweek, team)
    future_fixture_scores = [
        max(0.0, 5.0 - _fixture_difficulty(season_path, future_week, team))
        for future_week in range(gameweek, gameweek + 5)
    ]

    return {
        "form": _series(recent, "total_points").mean(),
        "points_per_game": _series(recent, "total_points").mean(),
        "minutes": minutes,
        "starts": float(starts),
        "recent_minutes_3": _series(recent_three, "minutes").sum(),
        "recent_starts_3": float((_series(recent_three, "minutes") >= 60).sum()),
        "recent_minutes_5": minutes,
        "recent_starts_5": float(starts),
        "expected_minutes": expected_minutes,
        "rotation_probability": rotation_probability,
        "goals": _series(recent, "goals_scored").sum(),
        "assists": _series(recent, "assists").sum(),
        "expected_goals": _series(recent, "expected_goals").sum(),
        "expected_assists": _series(recent, "expected_assists").sum(),
        "clean_sheets": _series(recent, "clean_sheets").sum(),
        "clean_sheet_probability": min(1.0, _series(recent, "clean_sheets").sum() / max(1.0, starts)),
        "price": price,
        "fixture_score": sum(future_fixture_scores) / len(future_fixture_scores),
        "fixture_congestion": fixture_congestion,
        "team_strength": _team_strength(season_path, team_name),
        "home_fixture_ratio": home_fixture_ratio,
        "current_gameweek": float(gameweek),
        "season_progress": min(1.0, float(gameweek) / 38),
        "position": float(position),
    }


def build_training_data(data_root: Path, seasons: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    histories = {season: _load_gameweeks(data_root / season) for season in seasons}
    profiles = {season: _season_profile(history) for season, history in histories.items()}
    for season in seasons:
        season_path = data_root / season
        history = histories[season]
        season_index = seasons.index(season)
        previous_profiles = profiles[seasons[season_index - 1]] if season_index else {}
        gameweeks = sorted(history["gameweek"].unique())
        for gameweek in gameweeks:
            if gameweek + 5 not in gameweeks:
                continue
            for player_id in history.loc[history["gameweek"] == gameweek, "element"].unique():
                features = _rolling_features(history, season_path, int(gameweek), int(player_id))
                if not features:
                    continue
                future = history[
                    (history["element"] == player_id)
                    & (history["gameweek"] > gameweek)
                    & (history["gameweek"] <= gameweek + 5)
                ]
                row = {
                    "season": season,
                    "gameweek": int(gameweek),
                    "player_id": int(player_id),
                    "team": _team_id(season_path, history.loc[history["element"] == player_id].iloc[-1].get("team")),
                    **features,
                }
                player_name = str(history.loc[history["element"] == player_id].iloc[-1].get("name", "")).lower().strip()
                row.update(previous_profiles.get(player_name, {
                    "previous_season_points": 0.0,
                    "previous_season_minutes": 0.0,
                    "previous_season_starts": 0.0,
                    "previous_season_points_per_90": 0.0,
                    "previous_season_expected_goals": 0.0,
                    "previous_season_expected_assists": 0.0,
                    "previous_season_clean_sheets": 0.0,
                }))
                row[TARGET_NAME] = float(_series(future, "total_points").sum())
                rows.append(row)

    columns = ["season", "gameweek", "player_id", "team", *FEATURE_NAMES, TARGET_NAME]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe historical rows for the FPL XGBoost model")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing season folders")
    parser.add_argument("--seasons", nargs="+", required=True, help="Season folders, e.g. 2020-21 2021-22")
    parser.add_argument("--output", type=Path, required=True, help="Output training CSV")
    args = parser.parse_args()

    training_data = build_training_data(args.data_root, args.seasons)
    if training_data.empty:
        raise SystemExit("No training rows were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(args.output, index=False)
    print(f"Wrote {len(training_data)} rows to {args.output}")


if __name__ == "__main__":
    main()
