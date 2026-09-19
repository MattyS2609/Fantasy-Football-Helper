from __future__ import annotations

from pathlib import Path
from typing import Any

from xgboost import XGBRegressor
from xgboost import DMatrix

from .recommendations import Player

FEATURE_NAMES = (
    "form",
    "points_per_game",
    "minutes",
    "starts",
    "recent_minutes_3",
    "recent_starts_3",
    "recent_minutes_5",
    "recent_starts_5",
    "expected_minutes",
    "rotation_probability",
    "goals",
    "assists",
    "expected_goals",
    "expected_assists",
    "clean_sheets",
    "clean_sheet_probability",
    "price",
    "fixture_score",
    "fixture_congestion",
    "team_strength",
    "home_fixture_ratio",
    "previous_season_points",
    "previous_season_minutes",
    "previous_season_starts",
    "previous_season_points_per_90",
    "previous_season_expected_goals",
    "previous_season_expected_assists",
    "previous_season_clean_sheets",
    "current_gameweek",
    "season_progress",
    "position",
)
TARGET_NAME = "target_points_next_5"


def player_features(
    player: Player,
    fixture_score: float,
    fixture_congestion: float = 0.0,
    home_fixture_ratio: float = 0.0,
    current_gameweek: int = 1,
    season_progress: float | None = None,
) -> list[float]:
    progress = season_progress if season_progress is not None else min(1.0, current_gameweek / 38)
    return [
        player.form,
        player.points_per_game,
        player.minutes,
        player.starts,
        player.recent_minutes_3,
        player.recent_starts_3,
        player.recent_minutes_5,
        player.recent_starts_5,
        player.expected_minutes,
        player.rotation_probability,
        player.goals,
        player.assists,
        player.expected_goals,
        player.expected_assists,
        player.clean_sheets,
        player.clean_sheet_probability,
        player.price,
        fixture_score,
        fixture_congestion,
        player.team_strength,
        home_fixture_ratio,
        player.previous_season_points,
        player.previous_season_minutes,
        player.previous_season_starts,
        player.previous_season_points_per_90,
        player.previous_season_expected_goals,
        player.previous_season_expected_assists,
        player.previous_season_clean_sheets,
        float(current_gameweek),
        progress,
        player.position,
    ]


class XGBoostPredictor:
    """Predicts a player's points over the next five gameweeks."""

    def __init__(self, estimator: XGBRegressor | None = None) -> None:
        self.estimator = estimator or XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        self.is_fitted = False

    def fit(self, records: list[dict[str, Any]]) -> XGBoostPredictor:
        if not records:
            raise ValueError("At least one historical training record is required")
        missing = set(FEATURE_NAMES) | {TARGET_NAME}
        missing -= records[0].keys()
        if missing:
            raise ValueError(f"Training records are missing columns: {sorted(missing)}")
        features = [[float(record[name]) for name in FEATURE_NAMES] for record in records]
        targets = [float(record[TARGET_NAME]) for record in records]
        self.estimator.fit(features, targets)
        self.is_fitted = True
        return self

    def predict(
        self,
        player: Player,
        fixture_score: float,
        fixture_congestion: float = 0.0,
        home_fixture_ratio: float = 0.0,
        current_gameweek: int = 1,
        season_progress: float | None = None,
    ) -> float:
        if not self.is_fitted:
            raise RuntimeError("XGBoostPredictor must be fitted or loaded before prediction")
        features = player_features(
            player,
            fixture_score,
            fixture_congestion,
            home_fixture_ratio,
            current_gameweek,
            season_progress,
        )
        return max(0.0, float(self.estimator.predict([features])[0]))

    def uncertainty(
        self,
        player: Player,
        fixture_score: float,
        fixture_congestion: float = 0.0,
        home_fixture_ratio: float = 0.0,
        current_gameweek: int = 1,
        season_progress: float | None = None,
    ) -> float:
        """Estimate model uncertainty from prediction variation across tree stages."""
        if not self.is_fitted:
            raise RuntimeError("XGBoostPredictor must be fitted or loaded before prediction")
        features = player_features(
            player,
            fixture_score,
            fixture_congestion,
            home_fixture_ratio,
            current_gameweek,
            season_progress,
        )
        booster = self.estimator.get_booster()
        stages = range(10, booster.num_boosted_rounds() + 1, 10)
        predictions = [
            float(booster.predict(DMatrix([features]), iteration_range=(0, stage))[0])
            for stage in stages
        ]
        return float(__import__("statistics").pstdev(predictions)) if len(predictions) > 1 else 0.0

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted XGBoostPredictor")
        self.estimator.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path) -> XGBoostPredictor:
        predictor = cls()
        predictor.estimator.load_model(str(path))
        predictor.is_fitted = True
        return predictor
