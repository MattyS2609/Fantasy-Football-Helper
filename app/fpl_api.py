from typing import Any

import httpx

BASE_URL = "https://fantasy.premierleague.com/api"


class FplApiError(RuntimeError):
    """Raised when the public FPL API cannot provide a response."""


class FplClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> Any:
        try:
            response = httpx.get(f"{self.base_url}/{path.lstrip('/')}", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise FplApiError(f"Unable to fetch FPL data: {path}") from exc

    def get_bootstrap(self) -> dict[str, Any]:
        return self._get("bootstrap-static/")

    def get_fixtures(self) -> list[dict[str, Any]]:
        return self._get("fixtures/")

    def get_picks(self, team_id: int, gameweek: int) -> dict[str, Any]:
        return self._get(f"entry/{team_id}/event/{gameweek}/picks/")

    def get_history(self, team_id: int) -> dict[str, Any]:
        return self._get(f"entry/{team_id}/history/")
