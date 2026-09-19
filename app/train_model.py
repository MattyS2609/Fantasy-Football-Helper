import csv
import sys
from pathlib import Path

from .model import FEATURE_NAMES, TARGET_NAME, XGBoostPredictor


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m app.train_model historical.csv models/fpl_xgb.json")

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    records = read_records(source)
    required = set(FEATURE_NAMES) | {TARGET_NAME}
    missing = required - records[0].keys() if records else required
    if missing:
        raise SystemExit(f"Training CSV is missing columns: {sorted(missing)}")

    XGBoostPredictor().fit(records).save(destination)
    print(f"Saved XGBoost model trained on {len(records)} rows to {destination}")


if __name__ == "__main__":
    main()
