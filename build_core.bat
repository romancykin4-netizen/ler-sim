@echo off
rem Builds the C++ core lersim\_lercore*.pyd (needs MSVC: Visual Studio or Build Tools, "Desktop development with C++")
cd /d "%~dp0"
call "%~dp0env_d.bat"
if not exist .venv call "%~dp0setup_env.bat" || exit /b 1
.venv\Scripts\python lercore\build.py || exit /b 1
.venv\Scripts\python -c "from lersim import _lercore; print('C++ core OK, threads:', _lercore.threads())" || exit /b 1
