from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from xgboost import DMatrix

from .model import FEATURE_NAMES, XGBoostPredictor


def feature_importance(model_path: Path) -> pd.DataFrame:
    predictor = XGBoostPredictor.load(model_path)
    scores = predictor.estimator.get_booster().get_score(importance_type="gain")
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        rows.append(
            {
                "rank": 0,
                "feature": name,
                "gain": float(scores.get(f"f{index}", 0.0)),
            }
        )
    report = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    report["rank"] = report.index + 1
    total_gain = report["gain"].sum()
    report["relative_importance"] = report["gain"] / total_gain if total_gain else 0.0
    return report


def temporal_importance(model_path: Path, data_path: Path) -> pd.DataFrame:
    predictor = XGBoostPredictor.load(model_path)
    data = pd.read_csv(data_path)
    required = {"gameweek", *FEATURE_NAMES}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Data CSV is missing columns: {sorted(missing)}")

    contributions = predictor.estimator.get_booster().predict(
        DMatrix(data[list(FEATURE_NAMES)].astype(float)), pred_contribs=True
    )[:, :-1]
    weeks = pd.to_numeric(data["gameweek"], errors="coerce")
    ranges = pd.cut(
        weeks,
        bins=[0, 5, 10, 20, 30, 38],
        labels=["GW1-5", "GW6-10", "GW11-20", "GW21-30", "GW31-38"],
    )
    rows = []
    for period in ranges.cat.categories:
        mask = ranges == period
        if not mask.any():
            continue
        importance = abs(contributions[mask.to_numpy()]).mean(axis=0)
        total = importance.sum()
        for feature, value in zip(FEATURE_NAMES, importance):
            rows.append(
                {
                    "period": str(period),
                    "feature": feature,
                    "mean_absolute_contribution": float(value),
                    "relative_importance": float(value / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["period", "mean_absolute_contribution"], ascending=[True, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect learned XGBoost feature importance")
    parser.add_argument("model", type=Path, help="Saved XGBoost model path")
    parser.add_argument("--top", type=int, default=20, help="Number of features to display")
    parser.add_argument("--output", type=Path, help="Optional CSV report path")
    parser.add_argument("--data", type=Path, help="Training CSV for week-range attribution")
    parser.add_argument("--temporal-output", type=Path, help="Optional week-range attribution CSV")
    args = parser.parse_args()

    report = feature_importance(args.model)
    display = report.head(max(1, args.top)).copy()
    display["gain"] = display["gain"].round(4)
    display["relative_importance"] = (display["relative_importance"] * 100).round(2)
    print(display.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(f"\nSaved full feature-importance report to {args.output}")
    if args.data:
        temporal = temporal_importance(args.model, args.data)
        for period, group in temporal.groupby("period", sort=False):
            print(f"\n{period}")
            display = group.head(max(1, args.top)).copy()
            display["mean_absolute_contribution"] = display["mean_absolute_contribution"].round(4)
            display["relative_importance"] = (display["relative_importance"] * 100).round(2)
            print(display.to_string(index=False))
        if args.temporal_output:
            args.temporal_output.parent.mkdir(parents=True, exist_ok=True)
            temporal.to_csv(args.temporal_output, index=False)
            print(f"\nSaved temporal feature-importance report to {args.temporal_output}")


if __name__ == "__main__":
    main()
