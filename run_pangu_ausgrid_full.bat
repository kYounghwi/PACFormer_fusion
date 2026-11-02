@echo off
setlocal

set "OPTIMIZER=%~1"
if "%OPTIMIZER%"=="" set "OPTIMIZER=adam"
if /I not "%OPTIMIZER%"=="adam" if /I not "%OPTIMIZER%"=="muon" (
  echo Usage: run_pangu_ausgrid_full.bat [adam^|muon]
  exit /b 2
)

cd /d "%~dp0"
python run_ausgrid_full.py --model pangu --optimizer %OPTIMIZER%
exit /b %errorlevel%
