@rem All temp/cache of pip, PyInstaller and matplotlib go to <project>\.cache on drive D
set "LOC=%~dp0.cache"
if not exist "%LOC%\tmp" mkdir "%LOC%\tmp"
set "TMP=%LOC%\tmp"
set "TEMP=%LOC%\tmp"
set "PIP_CACHE_DIR=%LOC%\pip"
set "PYINSTALLER_CONFIG_DIR=%LOC%\pyinstaller"
set "MPLCONFIGDIR=%LOC%\matplotlib"
