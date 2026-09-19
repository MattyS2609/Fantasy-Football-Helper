import os
import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .fpl_api import FplApiError, FplClient
from .model import XGBoostPredictor
from .recommendations import recommend_transfers

app = FastAPI(title="FPL Helper", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _current_gameweek(bootstrap: dict[str, Any]) -> int:
    events = bootstrap.get("events", [])
    current = next((event for event in events if event.get("is_current")), None)
    if current is None:
        current = next((event for event in events if event.get("is_next")), None)
    if current is None:
        raise HTTPException(status_code=503, detail="No current FPL gameweek is available")
    return int(current["id"])


@lru_cache(maxsize=1)
def _historical_predictor() -> XGBoostPredictor | None:
    model_path = os.getenv("FPL_MODEL_PATH") or str(Path(__file__).resolve().parents[1] / "models" / "fpl_xgb.json")
    if not os.path.exists(model_path):
        return None
    return XGBoostPredictor.load(model_path)


@lru_cache(maxsize=1)
def _player_profiles() -> dict[str, dict[str, float]]:
    profile_path = os.getenv("FPL_PLAYER_PROFILES_PATH", "data/player_profiles.csv")
    if not os.path.exists(profile_path):
        return {}
    with open(profile_path, newline="", encoding="utf-8") as file:
        return {
            str(row["player_name"]).lower().strip(): {
                key: float(value)
                for key, value in row.items()
                if key not in {"player_name", "season"} and value
            }
            for row in csv.DictReader(file)
        }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/recommendations/{team_id}")
def recommendations(
    team_id: int,
    gameweek: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    client = FplClient()
    try:
        bootstrap = client.get_bootstrap()
        current_gameweek = gameweek or _current_gameweek(bootstrap)
        fixtures = client.get_fixtures()
        picks_response = client.get_picks(team_id, current_gameweek)
        history = client.get_history(team_id)
    except FplApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    current = history.get("current", [])
    profiles = _player_profiles()
    elements = []
    for player in bootstrap.get("elements", []):
        profile = profiles.get(str(player.get("web_name", "")).lower().strip(), {})
        elements.append({**player, **profile})
    team_strengths = {
        int(team["id"]): (
            float(team.get("strength_overall_home", team.get("strength", 3)))
            + float(team.get("strength_overall_away", team.get("strength", 3)))
        ) / 2000
        for team in bootstrap.get("teams", [])
    }
    predictor = _historical_predictor()
    bank = float(current[-1].get("bank", 0)) / 10 if current else 0.0
    transfers = recommend_transfers(
        elements=elements,
        fixtures=fixtures,
        picks=picks_response.get("picks", []),
        bank=bank,
        current_gameweek=current_gameweek,
        predictor=predictor,
        team_strengths=team_strengths,
    )
    return {
        "team_id": team_id,
        "gameweek": current_gameweek,
        "bank": bank,
        "model": "xgboost" if predictor is not None else "heuristic",
        "recommendations": transfers,
    }
