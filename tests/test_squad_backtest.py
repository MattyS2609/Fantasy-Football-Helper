import pandas as pd

from app.evaluate_model import backtest_squads
from app.model import FEATURE_NAMES, TARGET_NAME


def test_squad_backtest_returns_no_transfer_control():
    rows = []
    for season in ["old", "test"]:
        for gameweek in [2, 3, 4, 5, 6, 7]:
            for player_id in range(1, 16):
                position = 3 if player_id <= 3 else 1 + player_id % 4
                team = 1 if player_id == 1 else 2 + player_id % 4
                points = 2 if player_id == 1 else 8 if player_id == 2 else 3
                row = {
                    "season": season,
                    "gameweek": gameweek,
                    "player_id": player_id,
                    "team": team,
                    TARGET_NAME: points if season == "test" else points - 1,
                }
                row.update({name: 1.0 for name in FEATURE_NAMES})
                row["position"] = position
                row["price"] = 5.0
                row["points_per_game"] = float(points)
                row["expected_minutes"] = 90.0
                rows.append(row)

    squads = pd.DataFrame(
        [
            {"season": "test", "gameweek": 2, "player_id": 1, "starting": 1, "selling_price": 5.0, "bank": 1.0},
            *[
                {"season": "test", "gameweek": 2, "player_id": player_id, "starting": 1, "selling_price": 5.0, "bank": 1.0}
                for player_id in range(2, 16)
            ],
        ]
    )

    report = backtest_squads(pd.DataFrame(rows), squads, "test")

    assert set(report["model"]) == {"XGBoost", "points-per-game baseline", "no transfer"}
    assert report.loc[report["model"] == "no transfer", "average_actual_gain"].iloc[0] == 0
