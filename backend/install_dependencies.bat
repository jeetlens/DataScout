@echo off
REM Quick Backend Dependencies Fix for DataScout

echo DataScout Backend - Installing Missing Dependencies
echo.

REM Check if we're in the backend directory
if not exist main.py (
    echo ERROR: main.py not found. Make sure you're in the backend directory.
    echo Current directory should be: C:\Users\jeetr\Desktop\Project\DataScout\backend
    pause
    exit /b 1
)

REM Check if virtual environment is activated
python -c "import sys; print('Virtual env active:' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'Virtual env NOT active'); exit(0)"

echo.
echo Installing core dependencies step by step...
echo.

REM Install core packages first
echo Step 1: Installing FastAPI and Uvicorn...
pip install fastapi uvicorn[standard]
if errorlevel 1 (
    echo ERROR: Failed to install FastAPI/Uvicorn
    pause
    exit /b 1
)

echo.
echo Step 2: Installing data processing packages...
pip install pandas numpy
if errorlevel 1 (
    echo ERROR: Failed to install pandas/numpy
    pause
    exit /b 1
)

echo.
echo Step 3: Installing scikit-learn (the correct way)...
pip install scikit-learn
if errorlevel 1 (
    echo ERROR: Failed to install scikit-learn
    pause
    exit /b 1
)

echo.
echo Step 4: Installing visualization packages...
pip install matplotlib seaborn plotly
if errorlevel 1 (
    echo ERROR: Failed to install visualization packages
    pause
    exit /b 1
)

echo.
echo Step 5: Installing additional packages...
pip install scipy openpyxl reportlab requests
if errorlevel 1 (
    echo ERROR: Failed to install additional packages
    pause
    exit /b 1
)

echo.
echo Step 6: Installing remaining requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some packages from requirements.txt may have failed
    echo But core packages are installed, server should start
)

echo.
echo ✅ Installation completed!
echo.
echo Testing imports...
python -c "
try:
    import fastapi, pandas, numpy, sklearn
    print('✅ All core imports successful!')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

echo.
echo Now try starting the server:
echo   python main.py
echo.
pause