# XGBoost Notes

The API uses the explainable scorer by default. To enable a trained historical model, create a CSV with these columns:

```text
form,points_per_game,minutes,starts,expected_minutes,rotation_probability,goals,assists,clean_sheets,price,fixture_score,position,target_points_next_5
```

Each row must contain only information available before the prediction gameweek. `target_points_next_5` is the player's actual points over the following five gameweeks.

## Train and enable the model

```powershell
New-Item -ItemType Directory -Force models
.\\venv\\Scripts\\python.exe -m app.train_model historical.csv models\\fpl_xgb.json
$env:FPL_MODEL_PATH = "$PWD\\models\\fpl_xgb.json"
.\\venv\\Scripts\\python.exe -m uvicorn app.main:app --reload
```

Use season-based validation before enabling the model in live recommendations. The model is loaded only when `FPL_MODEL_PATH` exists; otherwise the current heuristic remains active.

When XGBoost is enabled, its prediction is used directly throughout the season. The basic heuristic is only used as a fallback when no trained model is available. XGBoost receives both individual previous-season performance and `current_gameweek`/`season_progress`, allowing it to learn how historical performance should interact with season progress without a separate manual fade.

## Inspect feature importance

To see which inputs the trained model relies on most:

```powershell
.\\venv\\Scripts\\python.exe -m app.inspect_model `
\tmodels\\fpl_xgb.json `
\t--top 20 `
\t--output data\\feature_importance.csv
```

The report uses XGBoost gain importance. `relative_importance` is the percentage of total tree gain attributed to each feature. This describes what the model uses to make splits; it does not prove that a feature causes higher predicted points.

To compare learned feature influence at different stages of the season:

```powershell
.\\venv\\Scripts\\python.exe -m app.inspect_model `
\tmodels\\fpl_xgb.json `
\t--data data\\historical.csv `
\t--top 10 `
\t--temporal-output data\\feature_importance_by_period.csv
```

This uses XGBoost prediction contributions and reports mean absolute contribution for GW1-5, GW6-10, GW11-20, GW21-30, and GW31-38. Compare the `relative_importance` of previous-season features across these periods to see whether the trained model actually reduces their influence.

XGBoost also receives `current_gameweek` and `season_progress` as features. This lets it learn how the value of previous-season performance changes as more current-season evidence becomes available. The initial holdout result with these features was MAE `3.8873` and average transfer gain `6.7870`; the previous model scored MAE `3.8085` and average transfer gain `7.5970`, so this version should be treated as experimental until it is validated across more held-out seasons.

## Build the historical CSV

Download the historical repository, then point the builder at its `data` directory:

```powershell
git clone --depth 1 https://github.com/vaastav/Fantasy-Premier-League.git historical-data
.\\venv\\Scripts\\python.exe -m app.build_training_data `
\t--data-root historical-data\\data `
\t--seasons 2020-21 2021-22 2022-23 2023-24 2024-25 `
\t--output data\\historical.csv
```

The builder skips the opening gameweek of each season until prior-match features exist and skips the final five gameweeks because a complete target window is not available. Inspect `data\\historical.csv` before training the model.

## Evaluate before enabling

Hold out a complete season so the model is tested on data it did not train on:

```powershell
.\\venv\\Scripts\\python.exe -m app.evaluate_model `
\tdata\\historical.csv `
\t--test-season 2024-25 `
\t--report data\\evaluation_2024-25.csv
```

The evaluation compares XGBoost with a five-gameweek points-per-game baseline using mean absolute error, root mean squared error, and correlation. The first holdout run on `2024-25` produced MAE `3.7795` for XGBoost versus `4.0877` for the baseline, but transfer-level backtesting is still needed before treating that improvement as better FPL advice.

The same command also writes `data\\evaluation_2024-25_transfers.csv`. The initial transfer backtest covered `22,577` held-out player-swap cases: XGBoost achieved an average actual gain of `7.0338` points and a positive-transfer rate of `75.19%`, compared with `6.6624` points and `73.84%` for the baseline. This first backtest checks individual same-position swaps with a zero-bank assumption; it does not yet simulate complete squads, free transfers, hits, or the three-player-per-club rule.

For full squad-level backtesting, provide a CSV of historical squad snapshots with one row per squad player:

```text
season,gameweek,player_id,starting,selling_price,bank
2024-25,10,123,1,7.2,0.5
```

Run the squad simulation with:

```powershell
.\\venv\\Scripts\\python.exe -m app.evaluate_model `
\tdata\\historical.csv `
\t--test-season 2024-25 `
\t--squads data\\squad_snapshots.csv `
\t--report data\\evaluation_2024-25.csv
```

This compares XGBoost, the baseline, and a no-transfer strategy while enforcing position, budget, and three-player-per-club constraints. The historical repository does not include manager squad snapshots, so the squad CSV must come from an FPL manager export or a separately collected dataset.

Collect a snapshot of your current team after each gameweek with:

```powershell
.\\venv\\Scripts\\python.exe -m app.collect_squad_snapshot 6377026 `
\t--output data\\squad_snapshots.csv
```

The command appends 15 rows for the current gameweek. Run it once per gameweek, after making any transfers. Past seasons cannot be reconstructed from the public API, so these snapshots build the dataset for future backtests.
