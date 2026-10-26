@echo off
rem Builds dist\LER_sim\LER_sim.exe and dist\LER_sim_win64.zip (no Python needed on target PC)
cd /d "%~dp0"
call "%~dp0env_d.bat"
call "%~dp0setup_env.bat" || exit /b 1
call "%~dp0build_core.bat"
if errorlevel 1 (
  echo.
  echo WARNING: C++ core was not built, the program will use the Python core.
  echo          Install "Desktop development with C++" in Visual Studio Installer to enable it.
  echo.
)
.venv\Scripts\python -m pytest -q || exit /b 1
.venv\Scripts\python -m PyInstaller --noconfirm --clean --workpath "%LOC%\build" ler_sim.spec || exit /b 1
.venv\Scripts\python pack.py || exit /b 1
echo.
echo OK: dist\LER_sim\LER_sim.exe
echo     dist\LER_sim_win64.zip
