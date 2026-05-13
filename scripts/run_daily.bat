@echo off
REM Phase 24.18 — daily Omni_Stocks operations runner.
REM Schedule with Windows Task Scheduler at 18:30 IST (post-NSE-bhavcopy).
REM
REM Pipeline (production_runner -> daily.run_daily):
REM   mount drive  ->  ingest (prices/fundamentals/news/macro)
REM   features  ->  regime  ->  predict  ->  signals  ->  persist
REM
REM See docs/setup.md "Daily ops" for Task Scheduler install command.

setlocal

cd /d C:\Omni_Stocks
set PYTHONPATH=C:\Omni_Stocks

REM Ensure logs directory exists. Windows date format is locale-dependent;
REM use a safe ISO-ish stem (YYYYMMDD-HHMM) via wmic.
if not exist logs mkdir logs
for /f %%i in ('wmic os get LocalDateTime ^| find "."') do set _DT=%%i
set _STAMP=%_DT:~0,8%-%_DT:~8,4%

REM Activate venv and invoke the production runner. stdout+stderr go to a
REM dated log so Task Scheduler captures them even when run as SYSTEM.
"C:\Omni_Stocks\.venv\Scripts\python.exe" -m omni.orchestrator.production_runner ^
    > "logs\daily_%_STAMP%.log" 2>&1

set _RC=%ERRORLEVEL%
echo [run_daily] exit=%_RC% log=logs\daily_%_STAMP%.log

REM Surface non-zero exit so Task Scheduler can mark the run failed.
exit /b %_RC%
