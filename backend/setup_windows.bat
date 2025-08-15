@echo off
REM DataScout Backend Setup Script for Windows
REM This script sets up the Python environment and installs dependencies

echo Starting DataScout Backend Setup...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking version...
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.8+ is required
    python --version
    pause
    exit /b 1
)

echo Python version OK.
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully.
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated.
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

echo.

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    echo.
    echo Try installing manually:
    echo   pip install fastapi uvicorn pandas numpy matplotlib
    pause
    exit /b 1
)

echo.
echo ✅ Setup completed successfully!
echo.
echo To start the backend server:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run server: python main.py
echo.
echo To deactivate virtual environment later: deactivate
echo.
pause