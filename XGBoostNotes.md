# XGBoost Notes

## What is available on GitHub?

The GitHub repository contains the Python code needed to build and use the model, but it does not contain the trained model itself or the generated training data.

These items are intentionally ignored by Git:

- `.venv/`: a local Python virtual environment that must be created per machine.
- `historical-data/`: a locally cloned historical-data repository.
- `data/`: generated training, evaluation, and squad-snapshot CSV files.
- `models/`: generated model files such as `fpl_xgb.json`.

This keeps the repository small and avoids treating generated or machine-specific files as source code. A fresh clone uses the explainable heuristic scorer unless a local model is created and selected with `FPL_MODEL_PATH`.

## Option 1: Rebuild the model

Run these commands from the project root in PowerShell.

### 1. Create the local Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Download the historical source data

The training-data builder expects season folders containing the historical repository's gameweek CSV files:

```powershell
git clone --depth 1 https://github.com/vaastav/Fantasy-Premier-League.git historical-data
```

### 3. Build the training CSV

Choose the seasons to use and write the generated CSV under the ignored `data/` directory:

```powershell
New-Item -ItemType Directory -Force data
python -m app.build_training_data `
    --data-root historical-data\data `
    --seasons 2020-21 2021-22 2022-23 2023-24 2024-25 `
    --output data\historical.csv
```

The builder avoids rows without prior-match information and excludes the final five gameweeks when a complete five-gameweek target cannot be calculated. The resulting CSV must contain the model features and target used by `app.train_model`:

```text
form,points_per_game,minutes,starts,recent_minutes_3,recent_starts_3,recent_minutes_5,recent_starts_5,expected_minutes,rotation_probability,goals,assists,expected_goals,expected_assists,clean_sheets,clean_sheet_probability,price,fixture_score,fixture_congestion,team_strength,home_fixture_ratio,previous_season_points,previous_season_minutes,previous_season_starts,previous_season_points_per_90,previous_season_expected_goals,previous_season_expected_assists,previous_season_clean_sheets,current_gameweek,season_progress,position,target_points_next_5
```

### 4. Train the XGBoost model

Create the ignored `models/` directory and train a local model:

```powershell
New-Item -ItemType Directory -Force models
python -m app.train_model data\historical.csv models\fpl_xgb.json
```

### 5. Run the API with the model enabled

Set the model path for the current PowerShell session, then start the API:

```powershell
$env:FPL_MODEL_PATH = "$PWD\models\fpl_xgb.json"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

When `FPL_MODEL_PATH` is unset or points to a missing file, the API falls back to the default heuristic scorer.

## Validate before using

Evaluate against a held-out season before using the rebuilt model for recommendations:

```powershell
python -m app.evaluate_model `
    data\historical.csv `
    --test-season 2024-25 `
    --report data\evaluation_2024-25.csv
```

The evaluation output is generated locally under `data/`. Do not treat one holdout result as proof that the model gives better FPL advice; validate across additional seasons where possible.

To inspect model feature importance:

```powershell
python -m app.inspect_model `
    models\fpl_xgb.json `
    --top 20 `
    --output data\feature_importance.csv
```

## Option 2: Request a demo

Rebuilding requires downloading the historical source data and may not reproduce the exact model previously trained by the project owner. If you only need to see the existing model in operation, contact the project owner to request a demo or access to the existing local model. The model file is not available in the public GitHub repository.
