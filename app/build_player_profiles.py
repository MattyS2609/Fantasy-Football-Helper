from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .build_training_data import _load_gameweeks, _season_profile


def build_profiles(data_root: Path, season: str) -> pd.DataFrame:
    history = _load_gameweeks(data_root / season)
    profiles = _season_profile(history)
    return pd.DataFrame(
        [{"player_name": name, "season": season, **values} for name, values in profiles.items()]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build individual player previous-season profiles")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--season", required=True, help="Most recent completed season")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = build_profiles(args.data_root, args.season)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.output, index=False)
    print(f"Wrote {len(profiles)} player profiles to {args.output}")


if __name__ == "__main__":
    main()
