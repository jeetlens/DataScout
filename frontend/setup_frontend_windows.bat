@echo off
REM DataScout Frontend Setup Script for Windows
REM This script sets up Node.js environment and installs dependencies

echo Starting DataScout Frontend Setup...
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

echo Node.js found. Checking version...
node -e "const v=process.versions.node.split('.')[0]; process.exit(v>=18?0:1)"
if errorlevel 1 (
    echo ERROR: Node.js 18+ is required
    node --version
    pause
    exit /b 1
)

echo Node.js version OK.
echo.

REM Check if npm is available
npm --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm is not available
    pause
    exit /b 1
)

echo npm found.
echo.

REM Clean previous installation if it exists
if exist node_modules (
    echo Cleaning previous installation...
    rmdir /s /q node_modules
    if errorlevel 1 (
        echo Warning: Could not remove node_modules completely
    )
)

if exist package-lock.json (
    echo Removing old lock file...
    del package-lock.json
)

echo.
echo Installing dependencies... This may take a few minutes.
echo.

REM Clear npm cache first
echo Clearing npm cache...
npm cache clean --force

REM Install dependencies with verbose output
echo Running: npm install
npm install --verbose
if errorlevel 1 (
    echo.
    echo ERROR: npm install failed
    echo.
    echo Trying alternative installation methods...
    echo.
    
    REM Try with different registry
    echo Trying with npm registry...
    npm install --registry https://registry.npmjs.org/
    if errorlevel 1 (
        echo.
        echo ERROR: Installation failed with alternative registry too
        echo.
        echo Manual troubleshooting steps:
        echo   1. Run as Administrator
        echo   2. Check your internet connection
        echo   3. Try: npm install --force
        echo   4. Try: npm config set registry https://registry.npmjs.org/
        echo   5. Install globally: npm install -g vite
        pause
        exit /b 1
    )
)

echo.
echo Verifying installation...

REM Check if node_modules was created
if not exist node_modules (
    echo ERROR: node_modules directory was not created
    pause
    exit /b 1
)

REM Check if vite is available
echo Checking if Vite is available...
npx vite --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Vite not found via npx, trying global installation...
    npm install -g vite
    if errorlevel 1 (
        echo ERROR: Could not install Vite globally either
        pause
        exit /b 1
    )
)

echo.
echo ✅ Frontend setup completed successfully!
echo.
echo Dependencies installed:
npm list --depth=0

echo.
echo To start the development server:
echo   npm run dev
echo.
echo Or use: start_frontend.bat
echo.
echo If 'vite' is not recognized, try:
echo   npx vite
echo   or
echo   npm run dev -- --host 0.0.0.0
echo.
pause