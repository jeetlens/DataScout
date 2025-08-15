@echo off
REM Simple .env file creator for DataScout - No user input required

echo DataScout Backend - Simple .env File Creator
echo.

REM Check if we're in the backend directory
if not exist main.py (
    echo ERROR: main.py not found. Make sure you're in the backend directory.
    pause
    exit /b 1
)

echo Creating .env template file...
echo.

REM Create .env template
(
echo # DataScout Backend Environment Configuration
echo # REPLACE 'your_gemini_api_key_here' with your actual API key
echo GOOGLE_API_KEY=your_gemini_api_key_here
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

echo ✅ .env template file created!
echo.
echo ⚠️  IMPORTANT: You must edit the .env file and replace 'your_gemini_api_key_here' with your actual API key
echo.
echo To edit the file:
echo   notepad .env
echo.
echo Or use any text editor to replace:
echo   your_gemini_api_key_here
echo With your actual Gemini API key from: https://makersuite.google.com/app/apikey
echo.
echo After editing, you can start the server with:
echo   python main.py
echo.
pause