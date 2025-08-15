@echo off
REM Quick Fix for Vite Issue - DataScout Frontend

echo DataScout Frontend - Quick Fix for Vite Issue
echo.

echo Current directory: %CD%
echo.

REM Check if we're in the right directory
if not exist package.json (
    echo ERROR: package.json not found. Make sure you're in the frontend directory.
    echo Current directory should be: C:\Users\jeetr\Desktop\Project\DataScout\frontend
    pause
    exit /b 1
)

echo Found package.json. Proceeding with fix...
echo.

REM Step 1: Clear npm cache
echo Step 1: Clearing npm cache...
npm cache clean --force

REM Step 2: Remove node_modules and package-lock.json
echo Step 2: Removing old installation files...
if exist node_modules (
    echo Removing node_modules...
    rmdir /s /q node_modules
)
if exist package-lock.json (
    echo Removing package-lock.json...
    del package-lock.json
)

REM Step 3: Install dependencies
echo Step 3: Installing dependencies...
echo This may take a few minutes...
npm install
if errorlevel 1 (
    echo.
    echo npm install failed. Trying alternative approach...
    echo.
    
    REM Try installing with legacy peer deps
    npm install --legacy-peer-deps
    if errorlevel 1 (
        echo.
        echo Still failing. Trying with force flag...
        npm install --force
        if errorlevel 1 (
            echo.
            echo ERROR: All installation methods failed.
            echo Please check your internet connection and try running as Administrator.
            pause
            exit /b 1
        )
    )
)

REM Step 4: Verify Vite installation
echo.
echo Step 4: Verifying Vite installation...
if exist node_modules\.bin\vite.cmd (
    echo ✅ Vite found in node_modules\.bin\
) else (
    echo ❌ Vite not found in expected location
    echo Trying to install Vite specifically...
    npm install vite@latest
)

REM Step 5: Test Vite command
echo.
echo Step 5: Testing Vite command...
npx vite --version
if errorlevel 1 (
    echo npx vite failed, trying direct path...
    if exist node_modules\.bin\vite.cmd (
        node_modules\.bin\vite.cmd --version
        if not errorlevel 1 (
            echo ✅ Vite works with direct path
            echo You can start the server with: node_modules\.bin\vite.cmd
        )
    ) else (
        echo ❌ Vite still not working
        echo.
        echo Last resort: Installing Vite globally...
        npm install -g vite
    )
) else (
    echo ✅ npx vite works correctly
)

echo.
echo Fix attempt completed. Now try:
echo   npm run dev
echo.
echo If it still doesn't work, try:
echo   npx vite
echo   or
echo   node_modules\.bin\vite.cmd
echo.
pause