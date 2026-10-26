@echo off
rem Creates .venv on drive D and installs dependencies; nothing is cached on drive C
cd /d "%~dp0"
call "%~dp0env_d.bat"
if not exist .venv py -3 -m venv .venv || exit /b 1
.venv\Scripts\python -m pip uninstall -y PySide6 PySide6_Addons PySide6_WebEngine PySide6_Pdf >nul 2>&1
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt || exit /b 1
echo.
echo OK: environment in %~dp0.venv
