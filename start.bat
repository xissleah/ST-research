@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo   AI Search Start
echo ========================================
echo.
echo Current directory:
echo %cd%
echo.

REM ------------------------------------------------------------
REM 1. Check required project files
REM ------------------------------------------------------------

if not exist "run.py" (
    echo ERROR: run.py was not found.
    echo Put start.bat in the same folder as run.py.
    echo.
    pause
    exit /b 1
)

if not exist "index.html" (
    echo ERROR: index.html was not found.
    echo Put start.bat in the same folder as index.html.
    echo.
    pause
    exit /b 1
)

if not exist "settings.yml" (
    echo ERROR: settings.yml was not found.
    echo Put your configured settings.yml in this folder.
    echo.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM 2. Choose Python
REM ------------------------------------------------------------

set "PYTHON_EXE="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo ERROR: Python was not found.
    echo Please run install_dependence.bat first, or install Python manually.
    echo.
    pause
    exit /b 1
)

echo Python:
%PYTHON_EXE% --version
echo.

REM ------------------------------------------------------------
REM 3. Check Docker
REM ------------------------------------------------------------

where docker >nul 2>nul
if errorlevel 1 (
    echo ERROR: Docker was not found.
    echo Please install Docker Desktop first.
    echo.
    pause
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo ERROR: Docker is not running.
    echo Please start Docker Desktop first.
    echo.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM 4. Always recreate SearXNG with local settings.yml mounted
REM This guarantees the current settings.yml is used.
REM It removes only the container, not the downloaded image.
REM ------------------------------------------------------------

echo Recreating SearXNG container with local settings.yml ...
docker rm -f searxng >nul 2>nul

docker run -d ^
  --name searxng ^
  -p 18080:8080 ^
  -v "%cd%\settings.yml:/etc/searxng/settings.yml:ro" ^
  searxng/searxng:latest

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create SearXNG container.
    echo.
    pause
    exit /b 1
)

echo.
echo SearXNG frontend:
echo http://localhost:18080
echo.

REM ------------------------------------------------------------
REM 5. Wait briefly for SearXNG
REM ------------------------------------------------------------

echo Waiting for SearXNG to become ready ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0; $i -lt 20; $i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:18080/search?q=test&format=json' -TimeoutSec 2; if($r.StatusCode -ge 200){ $ok=$true; break } } catch { Start-Sleep -Seconds 1 } }; if($ok){ exit 0 } else { exit 1 }" >nul 2>nul

if errorlevel 1 (
    echo WARNING: SearXNG did not respond to the JSON test in time.
    echo The AI backend will still start, but search may fail until SearXNG is ready.
    echo.
) else (
    echo SearXNG is ready.
    echo.
)

REM ------------------------------------------------------------
REM 6. Start AI Search backend
REM ------------------------------------------------------------

echo Starting AI Search backend ...
echo.
%PYTHON_EXE% run.py

echo.
echo AI Search stopped.
pause
exit /b 0
