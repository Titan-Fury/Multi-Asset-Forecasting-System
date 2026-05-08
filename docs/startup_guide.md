# Omni_Stocks Startup Guide (Windows)

This guide gives a clean runbook for starting the app, data jobs, and checking logs.

## 1) One-time setup (if not done already)

```powershell
cd C:\Omni_Stocks
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Optional but recommended (if your raw/cold storage is on Drive):

```powershell
& "$env:USERPROFILE\OneDrive\Desktop\Mount-OmniDrive.bat"
```

## 2) Local services and URLs

### A) Dashboard app (Streamlit)

Run:

```powershell
cd C:\Omni_Stocks
.\.venv\Scripts\activate
streamlit run omni/dashboard/app.py
```

Open in browser:

- `http://localhost:8501`

Notes:

- This repo is app-first (Streamlit UI). There is no separate FastAPI/Flask backend server in `omni/` right now.

### B) LLM backend (Ollama, optional but recommended)

If Ollama is installed, start/verify it:

```powershell
ollama serve
```

Default local URL:

- `http://localhost:11434`

Quick check:

```powershell
curl http://localhost:11434/api/tags
```

## 3) Data + model pipeline commands

Run these from `C:\Omni_Stocks` with venv active.

### Step 1: Backfill market data lake

```powershell
python scripts/backfill_universe.py --mode incremental
```

Useful variants:

```powershell
python scripts/backfill_universe.py --mode full --asset-classes us,india --lookback-years 5
python scripts/backfill_universe.py --mode full --asset-classes us,india,crypto,fx --limit 30
```

### Step 2: Build pooled training set

```powershell
python scripts/build_training_set.py
```

### Step 3: Train universal model(s)

```powershell
python scripts/train_universal_model.py --horizons 1,5,21
```

### Step 4: Daily realized backtest / drift checks

```powershell
python scripts/daily_backtest.py
```

## 4) Log files and monitoring

## Main application log

File:

- `C:\Omni_Stocks\data\omni.log`

Live tail in PowerShell:

```powershell
Get-Content C:\Omni_Stocks\data\omni.log -Wait
```

Show latest 100 lines:

```powershell
Get-Content C:\Omni_Stocks\data\omni.log -Tail 100
```

Search for warnings/errors:

```powershell
Select-String -Path C:\Omni_Stocks\data\omni.log -Pattern "WARNING|ERROR"
```

## Backtest/drift outputs

Written by `scripts/daily_backtest.py` under:

- `C:\Omni_Stocks\data\lake\monitoring\realized_metrics.parquet`
- `C:\Omni_Stocks\data\lake\monitoring\realized_predictions.parquet`
- `C:\Omni_Stocks\data\lake\monitoring\alert_YYYY-MM-DD.txt` (if triggered)

## Model artifacts

Expected model files:

- `C:\Omni_Stocks\data\models\universal_1d.joblib`
- `C:\Omni_Stocks\data\models\universal_5d.joblib`
- `C:\Omni_Stocks\data\models\universal_21d.joblib`

## Training set artifact

- `C:\Omni_Stocks\data\lake\training\training_set.parquet`

## 5) Recommended startup sequence (daily)

1. Activate env:
   - `cd C:\Omni_Stocks`
   - `.\.venv\Scripts\activate`
2. Ensure optional dependencies are up:
   - Drive mount (if used)
   - `ollama serve` (if using local LLM route)
3. Refresh data:
   - `python scripts/backfill_universe.py --mode incremental`
4. (If needed) retrain:
   - `python scripts/build_training_set.py`
   - `python scripts/train_universal_model.py --horizons 1,5,21`
5. Start dashboard:
   - `streamlit run omni/dashboard/app.py`
6. Open:
   - `http://localhost:8501`
7. Keep logs open in another terminal:
   - `Get-Content C:\Omni_Stocks\data\omni.log -Wait

## 6) Fast troubleshooting

### App not opening on `localhost:8501`

- Check Streamlit process is running in terminal.
- Re-run: `streamlit run omni/dashboard/app.py`
- If port conflict, run:

```powershell
streamlit run omni/dashboard/app.py --server.port 8502
```

Then open `http://localhost:8502`.

### LLM explanation errors

- Check Ollama URL: `http://localhost:11434`
- Verify models are pulled (for example):
  - `ollama pull qwen2.5:7b-instruct-q5_K_M`
  - `ollama pull phi3.5:mini`
- Confirm required API keys in `.env` for external providers.

### "No prices fetched" for a ticker

- Check symbol format and market suffix (examples: `RELIANCE.NS`, `BTC-USD`, `EURUSD=X`).
- Inspect warnings in `data\omni.log` for `fetch_prices` / `lookup_ticker`.

## 7) Handy checks

Environment smoke test:

```powershell
python -c "from omni.utils.env import universe, llm_routes; print('tickers:', sum(len(v['tickers']) for v in universe()['asset_classes'].values())); print('llm tasks:', list(llm_routes()['tasks'].keys()))"
```

Quick data fetch smoke test:

```powershell
python -c "from omni.ingest.prices import fetch_prices; print(fetch_prices('us','AAPL',start='2024-01-01',end='2024-02-01').shape)"
```
`