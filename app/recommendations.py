from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Player:
    id: int
    name: str
    position: int
    team: int
    price: float
    form: float
    points_per_game: float
    minutes: int
    starts: int
    recent_minutes_3: float
    recent_starts_3: float
    recent_minutes_5: float
    recent_starts_5: float
    expected_goals: float
    expected_assists: float
    team_strength: float
    clean_sheet_probability: float
    previous_season_points: float
    previous_season_minutes: float
    previous_season_starts: float
    previous_season_points_per_90: float
    previous_season_expected_goals: float
    previous_season_expected_assists: float
    previous_season_clean_sheets: float
    status: str
    chance: int | None
    goals: int
    assists: int
    clean_sheets: int

    @property
    def availability_score(self) -> float:
        if self.status in {"i", "s", "u"}:
            return 0.0
        if self.chance is not None:
            return self.chance / 100
        return 1.0

    @property
    def expected_minutes(self) -> float:
        """Estimate minutes in the next match from recent starting involvement."""
        if self.availability_score == 0:
            return 0.0
        if self.minutes <= 0 or self.starts <= 0:
            return 45.0 * self.availability_score

        match_equivalents = self.minutes / 90
        start_probability = min(1.0, self.starts / max(1.0, match_equivalents))
        minutes_per_start = min(90.0, self.minutes / self.starts)
        return minutes_per_start * start_probability * self.availability_score

    @property
    def rotation_probability(self) -> float:
        return round(1.0 - self.expected_minutes / 90.0, 2)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_players(elements: list[dict[str, Any]], team_strengths: dict[int, float] | None = None) -> list[Player]:
    return [
        Player(
            id=int(item["id"]),
            name=str(item.get("web_name", item.get("first_name", "Unknown"))),
            position=int(item["element_type"]),
            team=int(item["team"]),
            price=_number(item.get("now_cost")) / 10,
            form=_number(item.get("form")),
            points_per_game=_number(item.get("points_per_game")),
            minutes=int(item.get("minutes") or 0),
            starts=int(item.get("starts") or 0),
            recent_minutes_3=float(item.get("recent_minutes_3", item.get("minutes", 0)) or 0),
            recent_starts_3=float(item.get("recent_starts_3", item.get("starts", 0)) or 0),
            recent_minutes_5=float(item.get("recent_minutes_5", item.get("minutes", 0)) or 0),
            recent_starts_5=float(item.get("recent_starts_5", item.get("starts", 0)) or 0),
            status=str(item.get("status", "a")),
            chance=item.get("chance_of_playing_next_round"),
            goals=int(item.get("goals_scored") or 0),
            assists=int(item.get("assists") or 0),
            clean_sheets=int(item.get("clean_sheets") or 0),
            expected_goals=_number(item.get("expected_goals")),
            expected_assists=_number(item.get("expected_assists")),
            team_strength=(team_strengths or {}).get(int(item["team"]), _number(item.get("team_strength"), 3.0)),
            clean_sheet_probability=_number(
                item.get("clean_sheet_probability"),
            ) or min(1.0, _number(item.get("clean_sheets")) / max(1.0, _number(item.get("starts")))),
            previous_season_points=_number(item.get("previous_season_points")),
            previous_season_minutes=_number(item.get("previous_season_minutes")),
            previous_season_starts=_number(item.get("previous_season_starts")),
            previous_season_points_per_90=_number(item.get("previous_season_points_per_90")),
            previous_season_expected_goals=_number(item.get("previous_season_expected_goals")),
            previous_season_expected_assists=_number(item.get("previous_season_expected_assists")),
            previous_season_clean_sheets=_number(item.get("previous_season_clean_sheets")),
        )
        for item in elements
    ]


def _fixture_score(player: Player, fixtures: list[dict[str, Any]], current_gameweek: int) -> float:
    upcoming = [
        fixture
        for fixture in fixtures
        if fixture.get("event") is not None
        and current_gameweek <= fixture["event"] < current_gameweek + 5
        and player.team in {fixture.get("team_h"), fixture.get("team_a")}
    ]
    if not upcoming:
        return 0.0

    difficulty = []
    for fixture in upcoming:
        difficulty.append(
            fixture["team_h_difficulty"]
            if fixture.get("team_h") == player.team
            else fixture["team_a_difficulty"]
        )
    return max(0.0, 5.0 - sum(difficulty) / len(difficulty))


def _fixture_difficulty(player: Player, fixtures: list[dict[str, Any]], current_gameweek: int) -> float:
    upcoming = [
        fixture for fixture in fixtures
        if fixture.get("event") is not None
        and current_gameweek <= fixture["event"] < current_gameweek + 5
        and player.team in {fixture.get("team_h"), fixture.get("team_a")}
    ]
    if not upcoming:
        return 3.0
    difficulty = [
        fixture["team_h_difficulty"]
        if fixture.get("team_h") == player.team
        else fixture["team_a_difficulty"]
        for fixture in upcoming
    ]
    return sum(difficulty) / len(difficulty)


def _primary_reason(
    outgoing: Player,
    incoming: Player,
    outgoing_score: float,
    incoming_score: float,
    outgoing_fixture_difficulty: float,
    incoming_fixture_difficulty: float,
) -> str:
    advantages = {
        "stronger projected output": incoming_score - outgoing_score,
        "better upcoming fixtures": outgoing_fixture_difficulty - incoming_fixture_difficulty,
        "more reliable expected minutes": incoming.expected_minutes - outgoing.expected_minutes,
        "greater attacking involvement": (
            incoming.expected_goals + incoming.expected_assists
            - outgoing.expected_goals - outgoing.expected_assists
        ),
    }
    reason = max(advantages, key=advantages.get)
    return reason.capitalize()


def _fixture_features(player: Player, fixtures: list[dict[str, Any]], current_gameweek: int) -> tuple[float, float, float]:
    upcoming = [
        fixture for fixture in fixtures
        if fixture.get("event") is not None
        and current_gameweek <= fixture["event"] < current_gameweek + 5
        and player.team in {fixture.get("team_h"), fixture.get("team_a")}
    ]
    if not upcoming:
        return 0.0, 0.0, 0.0
    home_count = sum(fixture.get("team_h") == player.team for fixture in upcoming)
    return len(upcoming) / 5, home_count / len(upcoming), player.team_strength


def _projected_points(
    player: Player,
    fixtures: list[dict[str, Any]],
    current_gameweek: int,
    predictor: Any = None,
) -> float:
    fixture_score = _fixture_score(player, fixtures, current_gameweek)
    congestion, home_ratio, _ = _fixture_features(player, fixtures, current_gameweek)
    attacking = min(player.expected_goals * 0.08 + player.expected_assists * 0.05, 1.5)
    heuristic_score = (
        player.points_per_game * 0.55
        + player.form * 0.25
        + fixture_score * 0.8
        + attacking
        + player.clean_sheet_probability * 0.5
        + player.team_strength * 0.05
        + congestion * 0.1
        + home_ratio * 0.1
    ) * (player.expected_minutes / 90)
    current_score = heuristic_score
    if predictor is not None:
        current_score = predictor.predict(
            player,
            fixture_score,
            congestion,
            home_ratio,
            current_gameweek,
            min(1.0, current_gameweek / 38),
        )

    return current_score


def _prediction_uncertainty(
    player: Player,
    fixtures: list[dict[str, Any]],
    current_gameweek: int,
    predictor: Any,
) -> float:
    if not hasattr(predictor, "uncertainty"):
        return 0.0
    fixture_score = _fixture_score(player, fixtures, current_gameweek)
    congestion, home_ratio, _ = _fixture_features(player, fixtures, current_gameweek)
    return predictor.uncertainty(
        player,
        fixture_score,
        congestion,
        home_ratio,
        current_gameweek,
        min(1.0, current_gameweek / 38),
    )


def recommend_transfers(
    elements: list[dict[str, Any]],
    fixtures: list[dict[str, Any]],
    picks: list[dict[str, Any]],
    bank: float = 0.0,
    current_gameweek: int = 1,
    limit: int = 5,
    predictor: Any = None,
    team_strengths: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    players = {player.id: player for player in parse_players(elements, team_strengths)}
    squad_ids = {int(pick["element"]) for pick in picks}
    squad = [players[player_id] for player_id in squad_ids if player_id in players]
    selling_prices = {}
    for pick in picks:
        player_id = int(pick["element"])
        selling_prices[player_id] = (
            _number(pick["selling_price"]) / 10
            if pick.get("selling_price") is not None
            else players[player_id].price
        )
    club_counts = {}
    for player in squad:
        club_counts[player.team] = club_counts.get(player.team, 0) + 1

    recommendations = []
    for outgoing in squad:
        available_budget = selling_prices.get(outgoing.id, outgoing.price) + bank
        outgoing_score = _projected_points(outgoing, fixtures, current_gameweek, predictor)
        for incoming in players.values():
            if incoming.id in squad_ids or incoming.position != outgoing.position:
                continue
            if incoming.price > available_budget or incoming.availability_score == 0:
                continue
            post_transfer_count = club_counts.get(incoming.team, 0) + (
                1 if incoming.team != outgoing.team else 0
            )
            if post_transfer_count > 3:
                continue

            gain = _projected_points(incoming, fixtures, current_gameweek, predictor) - outgoing_score
            if gain <= 0:
                continue
            incoming_fixture_difficulty = _fixture_difficulty(incoming, fixtures, current_gameweek)
            outgoing_fixture_difficulty = _fixture_difficulty(outgoing, fixtures, current_gameweek)
            incoming_score = _projected_points(incoming, fixtures, current_gameweek, predictor)
            incoming_uncertainty = _prediction_uncertainty(incoming, fixtures, current_gameweek, predictor) if predictor is not None else 0.0
            recommendations.append(
                {
                    "sell": outgoing.name,
                    "buy": incoming.name,
                    "sell_id": outgoing.id,
                    "buy_id": incoming.id,
                    "cost_change": round(incoming.price - selling_prices.get(outgoing.id, outgoing.price), 1),
                    "projected_gain": round(gain, 2),
                    "sell_predicted_points": round(outgoing_score, 2),
                    "buy_predicted_points": round(incoming_score, 2),
                    "buy_uncertainty": round(incoming_uncertainty, 2),
                    "buy_prediction_low": round(max(0.0, incoming_score - incoming_uncertainty), 2),
                    "buy_prediction_high": round(incoming_score + incoming_uncertainty, 2),
                    "uncertainty_level": "High" if incoming_uncertainty >= 4 else "Medium" if incoming_uncertainty >= 2 else "Low",
                    "sell_expected_minutes": round(outgoing.expected_minutes),
                    "buy_expected_minutes": round(incoming.expected_minutes),
                    "sell_fixture_difficulty": round(outgoing_fixture_difficulty, 2),
                    "buy_fixture_difficulty": round(incoming_fixture_difficulty, 2),
                    "buy_rotation_probability": incoming.rotation_probability,
                    "reason": _primary_reason(
                        outgoing,
                        incoming,
                        outgoing_score,
                        incoming_score,
                        outgoing_fixture_difficulty,
                        incoming_fixture_difficulty,
                    ),
                }
            )

    recommendations.sort(key=lambda item: item["projected_gain"], reverse=True)
    selected = []
    recommendations_by_outgoing: dict[int, int] = {}
    for recommendation in recommendations:
        outgoing_id = recommendation["sell_id"]
        if recommendations_by_outgoing.get(outgoing_id, 0) >= 3:
            continue
        selected.append(recommendation)
        recommendations_by_outgoing[outgoing_id] = recommendations_by_outgoing.get(outgoing_id, 0) + 1
        if len(selected) == limit:
            break
    return selected
