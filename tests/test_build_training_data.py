import pandas as pd

from app.build_training_data import build_training_data
from app.model import FEATURE_NAMES, TARGET_NAME


def test_builds_prior_features_and_next_five_target(tmp_path):
    season_path = tmp_path / "2024-25"
    gameweek_path = season_path / "gws"
    gameweek_path.mkdir(parents=True)

    for gameweek in range(1, 8):
        pd.DataFrame(
            [
                {
                    "element": 1,
                    "team": 1,
                    "position": 3,
                    "total_points": gameweek,
                    "minutes": 90,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "value": 70,
                }
            ]
        ).to_csv(gameweek_path / f"gw{gameweek}.csv", index=False)

    pd.DataFrame(
        [
            {"event": week, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4}
            for week in range(1, 8)
        ]
    ).to_csv(season_path / "fixtures.csv", index=False)

    result = build_training_data(tmp_path, ["2024-25"])

    assert list(result.columns) == ["season", "gameweek", "player_id", "team", *FEATURE_NAMES, TARGET_NAME]
    first = result.iloc[0]
    assert first["gameweek"] == 2
    assert first["form"] == 1
    assert first[TARGET_NAME] == sum(range(3, 8))
