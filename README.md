# FPL Helper

A first-pass FPL transfer recommendation API. It reads a public FPL team, evaluates affordable single-player replacements, and returns the five highest-scoring transfers over the next five gameweeks.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run tests

```powershell
python -m pytest
```

## Start the entire application

You can run both the Python API and the frontend app locally.

### 1) Start the backend API

From the project root:

```powershell
.
\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/recommendations/{team_id}

The team ID is the number in your FPL team URL.

### 2) Start the frontend

Open a second terminal, then run:

```powershell
cd frontend
npm install
npm run dev
```

The frontend should open at:
- http://localhost:3000

The frontend is designed to call the backend API from the local FastAPI server.

### 3) Optional: enable the trained ML model

The trained XGBoost model and its generated training data are not included in GitHub because they are local, generated artifacts rather than required source code. They are ignored by Git so the repository stays lightweight.

See [XGBoostNotes.md](XGBoostNotes.md) for instructions to rebuild a model, enable it locally, validate it, or request a demo of the existing model from the project owner. Without a local model, the API uses the default heuristic scorer.

### Notes

The recommendation score is deliberately simple and explainable. It uses recent form, points per game, fixture difficulty, attacking output, and availability. It does not yet account for captaincy, chips, transfer hits, wildcard rules, or multiple-transfer combinations.

