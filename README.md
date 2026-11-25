# NBA Player Performance Predictor

[Python](https://www.python.org/) · [Automated checks](https://github.com/zrose1211/nba-player-performance-predictor/actions) · [Source files](https://github.com/zrose1211/nba-player-performance-predictor/tree/main)

## Forecasting next-season scoring

A regression project that compares machine learning models against a simple, strong benchmark: predicting that a player's next-season points per game (PPG) will equal their current-season PPG.

**Status:** modeling pipeline and synthetic-data tests are provided. Full historical NBA evaluation is pending; no real-data accuracy or improvement is claimed.

## Data

The default loader downloads NBA Stats-derived season totals from the [nba_data archive](https://github.com/llimllib/nba_data) (see its [data dictionary](https://github.com/llimllib/nba_data/blob/main/docs/DATA_DICTIONARY.md)) for completed seasons ending 2010–2026, filters to regular-season rows, and converts counting statistics to per-game values. Each season is cached locally. The live NBA Stats API is an optional source (`python pipeline.py --source api`), but can time out on hosted runners. Alternatively, supply a CSV containing one league-wide row per player-season with columns:

`PLAYER_ID, season, AGE, GP, MIN, PTS, REB, AST, FGA, FTA, FG3A`

The season column is the integer ending year (2025 means 2024–25). Raw data are not committed. If the archive or API is unavailable, provide a compatible local CSV.

## Modeling and evaluation

- Predict next-season PPG for players with at least 20 games in the current season who return the following season.
- Join adjacent player-seasons by stable player ID; reject duplicate player-season rows.
- Use age, games played, minutes, scoring, rebounding, assists, shooting volume, prior scoring and year-over-year scoring change.
- Split by target season: training before 2022, validation in 2022, holdout beginning in 2023 and extending through the completed 2025–26 season. Never use next-season statistics as predictors.
- Compare ridge regression, random forest, histogram gradient boosting and the current-season persistence baseline using validation MAE. Keep the baseline if it wins.
- Refit the selected model using pre-test seasons and report holdout MAE and RMSE in points per game.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python pipeline.py
```

With a local dataset:
```bash
python pipeline.py --csv path/to/player_seasons.csv
```

Outputs: `reports/metrics.json`, `reports/predictions.csv`, and `reports/forecast.png`.

## Limitations

The target is conditional on returning to the league, so this does not predict retirement, injury-related absence, or total seasonal production. Current-season minimum games filter excludes some players. Trades, roster changes, injuries, league scoring trends and role changes can alter future performance. External data availability may vary. The archive is a third-party NBA Stats-derived dataset. This is a retrospective benchmark, not a betting tool.

## Automated evaluation

The [Portfolio checks workflow](https://github.com/zrose1211/nba-player-performance-predictor/actions/workflows/checks.yml) runs the test suite on GitHub Actions. Select **Run workflow** to regenerate the evaluation outputs; download the `evaluation-results` artifact from a successful run. The refresh uses external data sources; availability and results depend on the downloaded snapshot. The workflow does not overwrite previously published historical metrics.
