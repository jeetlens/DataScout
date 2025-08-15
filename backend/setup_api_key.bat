@echo off
REM DataScout Backend - API Key Setup Script

echo DataScout Backend - API Key Configuration
echo.

REM Check if we're in the backend directory
if not exist main.py (
    echo ERROR: main.py not found. Make sure you're in the backend directory.
    echo Current directory should be: C:\Users\jeetr\Desktop\Project\DataScout\backend
    pause
    exit /b 1
)

echo You are in the correct backend directory.
echo.

REM Check if .env file already exists
if exist .env (
    echo .env file already exists. Current contents:
    echo ----------------------------------------
    type .env
    echo ----------------------------------------
    echo.
    echo Do you want to update it? (y/n)
    set /p update=
    if /i not "%update%"=="y" (
        echo Keeping existing .env file.
        goto :test_key
    )
    echo.
    echo Backing up existing .env to .env.backup
    copy .env .env.backup
)

echo.
echo Please enter your Google Gemini API key:
echo (You can get one from: https://makersuite.google.com/app/apikey)
echo.
set /p api_key=API Key: 

if "%api_key%"=="" (
    echo ERROR: No API key provided.
    pause
    exit /b 1
)

echo.
echo Creating .env file with your API key...

REM Create .env file
(
echo # DataScout Backend Environment Configuration
echo # Google Gemini AI API Key
echo GOOGLE_API_KEY=%api_key%
echo.
echo # Development Settings
echo DEBUG=true
echo LOG_LEVEL=INFO
echo.
echo # CORS Settings for Frontend
echo CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173
echo.
echo # Optional: OpenAI API Key ^(if you have one^)
echo # OPENAI_API_KEY=your_openai_key_here
echo.
echo # File Upload Settings
echo MAX_FILE_SIZE=100MB
echo UPLOAD_DIR=./uploads
) > .env

echo ✅ .env file created successfully!
echo.

:test_key
echo Testing API key configuration...
echo.

REM Test if python-dotenv is installed
python -c "import dotenv; print('✅ python-dotenv available')" 2>nul
if errorlevel 1 (
    echo Installing python-dotenv...
    pip install python-dotenv
)

REM Test API key loading
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    print(f'✅ API key loaded successfully (starts with: {api_key[:10]}...)')
    print('✅ Ready to use Gemini AI features!')
else:
    print('❌ API key not found')
    print('Make sure the .env file contains: GOOGLE_API_KEY=your_key')
"

echo.
echo Configuration complete! Your .env file contains:
echo ----------------------------------------
type .env
echo ----------------------------------------
echo.
echo ⚠️  IMPORTANT: Keep your API key secure and never share it publicly!
echo.
echo Now you can start the backend server:
echo   python main.py
echo.
pause