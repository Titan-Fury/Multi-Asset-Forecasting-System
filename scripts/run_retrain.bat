@echo off
REM Phase 24.18 — weekly Omni_Stocks retrain runner.
REM Schedule with Windows Task Scheduler at Sunday 02:00 (post-week close).
REM
REM Pipeline (retrain_all.py):
REM   1. build_training_set.py       (refresh lake -> training/training_set.parquet)
REM   2. train_universal_model.py    (universal_{1,5,21}d.joblib + .meta.json)
REM   3. train stack ensemble        (stack_h{1,5,21}.joblib)
REM   4. train India specialist      (india_specialist.joblib)
REM
REM See dashboard "Scheduled tasks" panel for install/run/uninstall buttons.

setlocal

cd /d C:\Omni_Stocks
set PYTHONPATH=C:\Omni_Stocks
set PYTHONIOENCODING=utf-8

if not exist logs mkdir logs
for /f %%i in ('wmic os get LocalDateTime ^| find "."') do set _DT=%%i
set _STAMP=%_DT:~0,8%-%_DT:~8,4%

REM Step 1: Rebuild training set from the lake (catches any newly-added
REM tickers in config/universe.yaml).
"C:\Omni_Stocks\.venv\Scripts\python.exe" scripts\build_training_set.py ^
    > "logs\retrain_%_STAMP%_build.log" 2>&1
set _RC=%ERRORLEVEL%
if not "%_RC%"=="0" (
    echo [run_retrain] build_training_set failed rc=%_RC%
    exit /b %_RC%
)

REM Step 2-4: Universal + stack + India specialist via retrain_all.py.
"C:\Omni_Stocks\.venv\Scripts\python.exe" scripts\retrain_all.py --horizons 1,5,21 ^
    > "logs\retrain_%_STAMP%_models.log" 2>&1

set _RC=%ERRORLEVEL%
echo [run_retrain] exit=%_RC% logs=logs\retrain_%_STAMP%_*.log

REM Surface non-zero exit so Task Scheduler can mark the run failed.
exit /b %_RC%
