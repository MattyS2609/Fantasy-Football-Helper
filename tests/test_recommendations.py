from app.recommendations import parse_players, recommend_transfers
from app.recommendations import _projected_points


PLAYERS = [
    {
        "id": 1,
        "web_name": "CurrentMid",
        "element_type": 3,
        "team": 1,
        "now_cost": 70,
        "form": "3.0",
        "points_per_game": "4.0",
        "minutes": 900,
        "starts": 10,
        "status": "a",
        "goals_scored": 1,
        "assists": 1,
    },
    {
        "id": 2,
        "web_name": "BetterMid",
        "element_type": 3,
        "team": 2,
        "now_cost": 65,
        "form": "8.0",
        "points_per_game": "7.0",
        "minutes": 900,
        "starts": 10,
        "status": "a",
        "goals_scored": 5,
        "assists": 4,
    },
    {
        "id": 3,
        "web_name": "UnavailableMid",
        "element_type": 3,
        "team": 3,
        "now_cost": 60,
        "form": "10.0",
        "points_per_game": "10.0",
        "minutes": 0,
        "starts": 0,
        "status": "i",
        "goals_scored": 10,
    },
]

FIXTURES = [
    {
        "event": 1,
        "team_h": 2,
        "team_a": 4,
        "team_h_difficulty": 2,
        "team_a_difficulty": 4,
    }
]


def test_recommends_highest_scoring_affordable_replacement():
    result = recommend_transfers(
        elements=PLAYERS,
        fixtures=FIXTURES,
        picks=[{"element": 1, "selling_price": 70}],
        current_gameweek=1,
    )

    assert result[0]["sell"] == "CurrentMid"
    assert result[0]["buy"] == "BetterMid"
    assert result[0]["projected_gain"] > 0


def test_excludes_unavailable_players():
    result = recommend_transfers(
        elements=PLAYERS,
        fixtures=FIXTURES,
        picks=[{"element": 1, "selling_price": 70}],
        current_gameweek=1,
    )

    assert all(item["buy"] != "UnavailableMid" for item in result)


def test_uses_current_price_when_selling_price_is_missing():
    result = recommend_transfers(
        elements=PLAYERS,
        fixtures=FIXTURES,
        picks=[{"element": 1}],
        current_gameweek=1,
    )

    assert result[0]["buy"] == "BetterMid"


def test_expected_minutes_reflects_rotation_and_availability():
    players = parse_players(PLAYERS)

    assert players[0].expected_minutes == 90
    assert players[0].rotation_probability == 0
    assert players[2].expected_minutes == 0


def test_returns_at_most_three_recommendations_for_one_squad_player():
    replacements = [
        {
            "id": player_id,
            "web_name": f"Replacement{player_id}",
            "element_type": 3,
            "team": player_id,
            "now_cost": 65,
            "form": str(10 - player_id / 10),
            "points_per_game": "7.0",
            "minutes": 900,
            "starts": 10,
            "status": "a",
        }
        for player_id in range(4, 9)
    ]
    result = recommend_transfers(
        elements=PLAYERS + replacements,
        fixtures=FIXTURES,
        picks=[{"element": 1}],
        current_gameweek=1,
        limit=5,
    )

    assert len(result) == 3
    assert {item["sell_id"] for item in result} == {1}


class FixedPredictor:
    def predict(
        self,
        player,
        fixture_score,
        fixture_congestion,
        home_fixture_ratio,
        current_gameweek,
        season_progress,
    ):
        return 100.0


def test_xgboost_remains_active_at_each_stage_of_season():
    player = parse_players([{**PLAYERS[0], "previous_season_points": 180, "previous_season_minutes": 2700, "previous_season_points_per_90": 6.0}])[0]
    early = _projected_points(player, FIXTURES, 1, FixedPredictor())
    later = _projected_points(player, FIXTURES, 6, FixedPredictor())

    assert early == 100
    assert later == 100
