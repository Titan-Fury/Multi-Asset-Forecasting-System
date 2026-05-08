# Manual setup

This document collects every step that must be done manually on the host
machine. Code-level setup (Python package install, configs) is automated by
`pip install -e .` and the YAMLs in `config/`.

## 1. Python 3.11+

Install from python.org or via winget:

```powershell
winget install -e --id Python.Python.3.11
```

Verify:

```powershell
python --version    # >= 3.11
```

Then create a venv and install Omni_Stocks in editable mode:

```powershell
cd C:\Omni_Stocks
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

## 2. WinFsp + rclone (Google Drive mount)

Already done if Phase 0 of the build was completed. Re-cap:

1. Install WinFsp from <https://winfsp.dev/rel/>. Verify the
   "WinFsp.Launcher" service is `Running` in Services.
2. Download rclone from <https://rclone.org/downloads/>, extract to
   `C:\rclone\`, add to user PATH.
3. Create a custom OAuth Client ID in Google Cloud Console (Drive scope),
   then `rclone config` -> create remote named `gdrive` of type `drive` with
   that client ID/secret.
4. Test: `rclone lsd gdrive:` should list your Drive folders.
5. Mount via `Mount-OmniDrive.bat` on the desktop (creates `Z:\`). Confirm
   `Z:\` shows up in Explorer.

## 3. Ollama (local LLM)

Install: <https://ollama.com/download/windows>

Pull the two models the router expects:

```powershell
ollama pull qwen2.5:7b-instruct-q5_K_M
ollama pull phi3.5:mini
```

Smoke test:

```powershell
ollama run qwen2.5:7b-instruct-q5_K_M "Reply with the word OK"
```

Ollama listens on `http://localhost:11434` by default -- the value already
set in `.env` as `OLLAMA_BASE_URL`.

## 4. OpenRouter (free-tier remote LLM)

1. Sign up at <https://openrouter.ai>.
2. Create an API key.
3. Paste it into `.env` as `OPENROUTER_API_KEY=sk-or-...`.

The router enforces 18rpm / 180rpd locally to stay safely under the free
tier limits (the actual limit is 20rpm/200rpd).

## 5. NewsAPI

1. Sign up at <https://newsapi.org/register> (free).
2. Paste the key into `.env` as `NEWSAPI_KEY=...`.

Free tier: 100 requests/day. The ingest module respects this; Phase 9 will
budget queries across the universe.

## 6. FRED (macro)

1. Request an API key at <https://fred.stlouisfed.org/docs/api/api_key.html>.
2. Paste it into `.env` as `FRED_API_KEY=...`.

No rate limit issues at our usage level.

## 7. Alpha Vantage (optional, US fallback)

If you want a yfinance fallback for US prices:

1. Get a free key at <https://www.alphavantage.co/support/#api-key>.
2. Paste into `.env` as `ALPHA_VANTAGE_KEY=...`.

Currently unused in v1 (yfinance has been reliable enough).

## 8. Finnhub (optional, US fundamentals fallback)

1. Free key at <https://finnhub.io/register>.
2. Paste into `.env` as `FINNHUB_KEY=...`.

Unused in v1.

## 9. Snowflake Cortex

The connection name `IU95928` is already set in `.env`. Cortex auth comes
from `~/.snowflake/connections.toml` (or wherever your Snowflake CLI/connector
config lives). To verify:

```powershell
python -c "import snowflake.connector as c; conn=c.connect(connection_name='IU95928'); cur=conn.cursor(); cur.execute(\"SELECT SNOWFLAKE.CORTEX.SENTIMENT('great quarter')\"); print(cur.fetchone()); conn.close()"
```

Expected: a positive float close to 1.0.

## 10. Reddit (deferred to v2)

`sources.yaml` has `social.reddit.enabled: false`. To turn on later:

1. Create a script-type app at <https://www.reddit.com/prefs/apps>.
2. Fill `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` in `.env`.
3. Set `social.reddit.enabled: true` in `sources.yaml`.

## 11. Optional: lock down `rclone.conf`

By default `rclone.conf` lives at `%APPDATA%\rclone\rclone.conf`. To restrict
to your user only (NTFS ACL):

```powershell
$cfg = "$env:APPDATA\rclone\rclone.conf"
icacls $cfg /inheritance:r
icacls $cfg /grant:r "$env:USERNAME:(R,W)"
```

Verify:

```powershell
icacls $cfg
```

Should show only your user.

## Final smoke test

```powershell
cd C:\Omni_Stocks
.\.venv\Scripts\activate
python -c "from omni.utils.env import universe, llm_routes; print('tickers:', sum(len(v['tickers']) for v in universe()['asset_classes'].values())); print('llm tasks:', list(llm_routes()['tasks'].keys()))"
```

Expected output:
```
tickers: 64
llm tasks: ['sentiment_per_article', 'theme_extraction', 'bulk_classification', 'daily_market_synthesis', 'signal_explanation', 'cross_validation']
```
