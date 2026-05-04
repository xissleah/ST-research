@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   AI Search Installer
echo ========================================
echo.
echo Current directory: %cd%
echo.

where python >nul 2>nul
if errorlevel 1 goto NO_PYTHON

if not exist "dependence.txt" goto NO_DEP

echo Python version:
python --version
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 goto VENV_FAIL
)

echo.
echo Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo.

echo ========================================
echo   Install dependencies
echo ========================================
echo.

for /f "usebackq delims=" %%P in ("dependence.txt") do call :ASK_INSTALL "%%P"

echo.
echo ========================================
echo   Model download
echo ========================================
echo.

choice /M "Download Qwen3.5-2B model"
if errorlevel 2 goto DONE

".venv\Scripts\python.exe" -c "import modelscope" >nul 2>nul
if errorlevel 1 (
    echo modelscope is required for model download.
    choice /M "Install modelscope now"
    if errorlevel 2 goto DONE
    ".venv\Scripts\python.exe" -m pip install modelscope
    if errorlevel 1 goto MODELSCOPE_FAIL
)

set "MODEL_ID=Qwen/Qwen3.5-2B"
set "MODEL_PATH="

if exist "cfg.txt" (
    for /f "usebackq tokens=1,* delims==" %%A in ("cfg.txt") do (
        if /I "%%A"=="MODEL_PATH" set "MODEL_PATH=%%B"
    )
)

if "%MODEL_PATH%"=="" set "MODEL_PATH=%cd%\model\Qwen3.5-2B"
set "MODEL_PATH=%MODEL_PATH:"=%"

echo.
echo Model ID: %MODEL_ID%
echo Target directory: %MODEL_PATH%
echo.

if not exist "%MODEL_PATH%" mkdir "%MODEL_PATH%"

set "MODEL_ID_ENV=%MODEL_ID%"
set "MODEL_PATH_ENV=%MODEL_PATH%"

".venv\Scripts\python.exe" -c "import os; from modelscope import snapshot_download; mid=os.environ['MODEL_ID_ENV']; out=os.environ['MODEL_PATH_ENV']; print('Model ID:', mid); print('Target:', out); snapshot_download(mid, local_dir=out); print('Done:', out)"
if errorlevel 1 goto MODEL_FAIL

goto DONE

:ASK_INSTALL
set "PKG=%~1"
if "%PKG%"=="" goto :eof
echo.
choice /M "Install %PKG%"
if errorlevel 2 (
    echo Skipped: %PKG%
    goto :eof
)
echo Installing: %PKG%
".venv\Scripts\python.exe" -m pip install "%PKG%"
if errorlevel 1 goto PKG_FAIL
echo Installed: %PKG%
goto :eof

:NO_PYTHON
echo ERROR: Python was not found.
echo Install Python 3.10 or 3.11 and enable "Add Python to PATH".
goto FAIL

:NO_DEP
echo ERROR: dependence.txt was not found in this folder.
goto FAIL

:VENV_FAIL
echo ERROR: failed to create .venv.
goto FAIL

:PKG_FAIL
echo ERROR: failed to install dependency: %PKG%
echo You can retry later or install manually:
echo .venv\Scripts\python.exe -m pip install "%PKG%"
goto FAIL

:MODELSCOPE_FAIL
echo ERROR: failed to install modelscope.
goto FAIL

:MODEL_FAIL
echo ERROR: model download failed.
echo Possible reasons:
echo 1. Network cannot access ModelScope.
echo 2. Model ID is unavailable.
echo 3. Not enough disk space.
goto FAIL

:DONE
echo.
echo ========================================
echo   Installer finished
echo ========================================
echo.
echo You can now run start.bat.
pause
exit /b 0

:FAIL
echo.
echo ========================================
echo   Installer stopped
echo ========================================
echo.
pause
exit /b 1
