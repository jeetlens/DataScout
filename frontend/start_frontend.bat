@echo off
REM DataScout Frontend Startup Script for Windows

echo Starting DataScout Frontend Development Server...
echo.

REM Check if node_modules exists
if not exist node_modules (
    echo Dependencies not found. Running setup first...
    call setup_frontend_windows.bat
    if errorlevel 1 (
        echo Setup failed. Please check the error above.
        pause
        exit /b 1
    )
)

REM Check if Vite is available
npx vite --version >nul 2>&1
if errorlevel 1 (
    echo Vite not found. Installing dependencies...
    npm install
    if errorlevel 1 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo Starting development server...
echo Frontend will be available at: http://localhost:5173
echo Press Ctrl+C to stop the server
echo.

REM Start the development server
npm run dev
if errorlevel 1 (
    echo.
    echo Failed to start development server.
    echo.
    echo Troubleshooting:
    echo   1. Make sure Node.js 18+ is installed
    echo   2. Try: npm install
    echo   3. Try: npx vite
    echo   4. Check if port 5173 is available
    pause
    exit /b 1
)

pause