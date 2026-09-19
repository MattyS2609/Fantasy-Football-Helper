from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .model import FEATURE_NAMES, TARGET_NAME, XGBoostPredictor


def _report(name: str, actual: pd.Series, predicted: list[float]) -> dict[str, float | str]:
    return {
        "model": name,
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "correlation": actual.corr(pd.Series(predicted, index=actual.index)),
    }


def evaluate(data: pd.DataFrame, test_season: str) -> pd.DataFrame:
    if "season" not in data:
        raise ValueError("Training CSV must include a season column")
    train = data[data["season"] != test_season].copy()
    test = data[data["season"] == test_season].copy()
    if train.empty or test.empty:
        raise ValueError("Both training and test rows are required")

    predictor = XGBoostPredictor().fit(train.to_dict("records"))
    x_test = test[list(FEATURE_NAMES)].astype(float).values.tolist()
    xgb_predictions = predictor.estimator.predict(x_test).clip(min=0)

    baseline_predictions = (
        test["points_per_game"].astype(float)
        * 5
        * test["expected_minutes"].astype(float)
        / 90
    ).tolist()
    actual = test[TARGET_NAME].astype(float)
    return pd.DataFrame(
        [
            _report("points-per-game baseline", actual, baseline_predictions),
            _report("XGBoost", actual, xgb_predictions.tolist()),
        ]
    )


def backtest_transfers(data: pd.DataFrame, test_season: str, top_n: int = 5) -> pd.DataFrame:
    """Measure actual future points from the best-ranked transfer candidates."""
    train = data[data["season"] != test_season].copy()
    test = data[data["season"] == test_season].copy()
    predictor = XGBoostPredictor().fit(train.to_dict("records"))
    test_features = test[list(FEATURE_NAMES)].astype(float).values.tolist()
    test["xgb_prediction"] = predictor.estimator.predict(test_features).clip(min=0)
    test["baseline_prediction"] = (
        test["points_per_game"].astype(float) * 5
        * test["expected_minutes"].astype(float) / 90
    )

    results: list[dict[str, float | str]] = []
    for (season, gameweek), gameweek_rows in test.groupby(["season", "gameweek"]):
        for _, outgoing in gameweek_rows.iterrows():
            candidates = gameweek_rows[
                (gameweek_rows["position"] == outgoing["position"])
                & (gameweek_rows["player_id"] != outgoing["player_id"])
                & (gameweek_rows["price"] <= outgoing["price"])
            ].copy()
            if candidates.empty:
                continue
            outgoing_actual = float(outgoing[TARGET_NAME])
            for model_name, prediction_column in (
                ("points-per-game baseline", "baseline_prediction"),
                ("XGBoost", "xgb_prediction"),
            ):
                candidates["predicted_gain"] = candidates[prediction_column] - outgoing[prediction_column]
                selected = candidates.nlargest(top_n, "predicted_gain")
                actual_gains = selected[TARGET_NAME] - outgoing_actual
                results.append(
                    {
                        "model": model_name,
                        "season": str(season),
                        "gameweek": float(gameweek),
                        "outgoing_player_count": 1.0,
                        "recommended_count": float(len(selected)),
                        "average_actual_gain": float(actual_gains.mean()),
                        "best_actual_gain": float(actual_gains.max()),
                        "positive_transfer_rate": float((actual_gains > 0).mean()),
                    }
                )

    if not results:
        raise ValueError("No eligible transfer pairs were found for the held-out season")
    report = pd.DataFrame(results)
    return (
        report.groupby("model", as_index=False)
        .agg(
            test_season=("season", "first"),
            transfer_cases=("outgoing_player_count", "sum"),
            average_actual_gain=("average_actual_gain", "mean"),
            best_actual_gain=("best_actual_gain", "mean"),
            positive_transfer_rate=("positive_transfer_rate", "mean"),
        )
    )


def backtest_squads(
    data: pd.DataFrame,
    squads: pd.DataFrame,
    test_season: str,
) -> pd.DataFrame:
    """Backtest one legal transfer from historical 15-player squad snapshots."""
    required_squad = {"season", "gameweek", "player_id", "starting", "selling_price", "bank"}
    missing = required_squad - set(squads.columns)
    if missing:
        raise ValueError(f"Squad CSV is missing columns: {sorted(missing)}")
    train = data[data["season"] != test_season].copy()
    test = data[data["season"] == test_season].copy()
    predictor = XGBoostPredictor().fit(train.to_dict("records"))
    test["xgb_prediction"] = predictor.estimator.predict(
        test[list(FEATURE_NAMES)].astype(float).values.tolist()
    ).clip(min=0)
    test["baseline_prediction"] = test["points_per_game"] * 5 * test["expected_minutes"] / 90
    test_lookup = test.set_index(["season", "gameweek", "player_id"])
    rows: list[dict[str, float | str]] = []

    for (season, gameweek), snapshot in squads[squads["season"] == test_season].groupby(["season", "gameweek"]):
        squad_ids = set(snapshot["player_id"].astype(int))
        snapshot = snapshot[snapshot["player_id"].astype(int).isin(test_lookup.index.get_level_values("player_id"))]
        if len(snapshot) < 11:
            continue
        current = test[
            (test["season"] == season)
            & (test["gameweek"] == gameweek)
            & (test["player_id"].isin(squad_ids))
        ].copy()
        if current.empty:
            continue
        club_counts = current.groupby("team").size().to_dict() if "team" in current else {}
        for model_name, prediction_column in (
            ("points-per-game baseline", "baseline_prediction"),
            ("XGBoost", "xgb_prediction"),
        ):
            best = None
            for _, outgoing in current.iterrows():
                budget = float(snapshot.loc[snapshot["player_id"] == outgoing["player_id"], "selling_price"].iloc[0]) + float(snapshot["bank"].iloc[0])
                candidates = test[
                    (test["season"] == season)
                    & (test["gameweek"] == gameweek)
                    & (~test["player_id"].isin(squad_ids))
                    & (test["position"] == outgoing["position"])
                    & (test["price"] <= budget)
                    & (test[prediction_column] > outgoing[prediction_column])
                ].copy()
                for _, incoming in candidates.iterrows():
                    if "team" in incoming and club_counts.get(incoming["team"], 0) + 1 > 3:
                        continue
                    predicted_gain = float(incoming[prediction_column] - outgoing[prediction_column])
                    if best is None or predicted_gain > best["predicted_gain"]:
                        best = {"predicted_gain": predicted_gain, "outgoing": outgoing, "incoming": incoming}
            if best is None:
                actual_gain = 0.0
            else:
                actual_gain = float(best["incoming"][TARGET_NAME] - best["outgoing"][TARGET_NAME])
            rows.append({"model": model_name, "season": str(season), "gameweek": float(gameweek), "actual_gain": actual_gain})
        rows.append({"model": "no transfer", "season": str(season), "gameweek": float(gameweek), "actual_gain": 0.0})

    if not rows:
        raise ValueError("No valid squad snapshots were found for the held-out season")
    report = pd.DataFrame(rows)
    return report.groupby("model", as_index=False).agg(
        test_season=("season", "first"),
        squad_cases=("gameweek", "count"),
        average_actual_gain=("actual_gain", "mean"),
        positive_gain_rate=("actual_gain", lambda values: (values > 0).mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XGBoost on an unseen FPL season")
    parser.add_argument("csv", type=Path, help="Historical training CSV")
    parser.add_argument("--test-season", required=True, help="Season held out for testing")
    parser.add_argument("--report", type=Path, help="Optional CSV report path")
    parser.add_argument("--squads", type=Path, help="Optional historical squad snapshots CSV")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    missing = (set(FEATURE_NAMES) | {TARGET_NAME, "season"}) - set(data.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")
    report = evaluate(data, args.test_season)
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    transfer_report = backtest_transfers(data, args.test_season)
    print("\nTransfer backtest")
    print(transfer_report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    squad_report = None
    if args.squads:
        squad_report = backtest_squads(data, pd.read_csv(args.squads), args.test_season)
        print("\nSquad backtest")
        print(squad_report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report, index=False)
        transfer_report.to_csv(args.report.with_name(f"{args.report.stem}_transfers.csv"), index=False)
        if squad_report is not None:
            squad_report.to_csv(args.report.with_name(f"{args.report.stem}_squads.csv"), index=False)


if __name__ == "__main__":
    main()
