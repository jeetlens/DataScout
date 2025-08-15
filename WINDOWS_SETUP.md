# 🪟 DataScout Windows Installation Guide

## Quick Setup for Windows Environment

### Prerequisites
1. **Python 3.8+** - Download from [python.org](https://python.org)
2. **Git** (optional) - Download from [git-scm.com](https://git-scm.com)

### Step-by-Step Installation

#### 1. Open Command Prompt or PowerShell
```cmd
# Navigate to the backend directory
cd C:\Users\jeetr\Desktop\Project\DataScout\backend
```

#### 2. Run the Windows Setup Script
```cmd
# Run the automated setup script
setup_windows.bat
```

**Or manually:**

#### 3. Manual Setup (if script fails)
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

#### 4. Set Environment Variables (Optional)
Create a `.env` file in the backend directory:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
DEBUG=true
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

#### 5. Start the Backend Server
```cmd
# Make sure virtual environment is activated
venv\Scripts\activate

# Start the server
python main.py
```

The server should start at: http://localhost:8000

### Troubleshooting Common Windows Issues

#### Issue 1: "Python command not found"
**Solution:**
- Make sure Python is installed and added to PATH
- Use `py` instead of `python` on some Windows systems
- Reinstall Python with "Add to PATH" option checked

#### Issue 2: "Virtual environment activation fails"
**Solution:**
```cmd
# Try PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use the full path
C:\path\to\project\venv\Scripts\activate.bat
```

#### Issue 3: "WeasyPrint installation fails"
**Solution:**
```cmd
# Install Visual C++ Build Tools first
# Or skip WeasyPrint and use reportlab instead:
pip install reportlab
```

#### Issue 4: "ModuleNotFoundError for pandas/numpy"
**Solution:**
```cmd
# Install specific packages manually
pip install pandas numpy matplotlib fastapi uvicorn

# Then try requirements.txt again
pip install -r requirements.txt
```

#### Issue 5: "Permission denied errors"
**Solution:**
- Run Command Prompt as Administrator
- Or install in user directory:
```cmd
pip install --user -r requirements.txt
```

### Minimal Installation (if full requirements fail)
```cmd
# Core packages only
pip install fastapi uvicorn pandas numpy matplotlib plotly

# Then try the server
python main.py
```

### Testing the Installation
1. **Backend Health Check:**
   - Open browser: http://localhost:8000/health
   - Should return: `{"status": "healthy"}`

2. **API Documentation:**
   - Open browser: http://localhost:8000/docs
   - Should show FastAPI documentation

3. **Test Import:**
   ```cmd
   python -c "import fastapi, pandas, numpy; print('All imports OK')"
   ```

### Next Steps After Backend Setup
1. Set up frontend (navigate to `../frontend`)
2. Install Node.js dependencies: `npm install`
3. Start frontend: `npm run dev`
4. Access application: http://localhost:5173

### Environment Variables for Windows
Create `.env` file with:
```env
# Required for AI features
GOOGLE_API_KEY=your_api_key_here

# Development settings
DEBUG=true
LOG_LEVEL=INFO

# CORS settings
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173

# File upload settings
MAX_FILE_SIZE=100MB
UPLOAD_DIR=./uploads
```

### Performance Tips for Windows
1. **Use SSD storage** for better file I/O performance
2. **Exclude project folder** from Windows Defender real-time scanning
3. **Use Windows Terminal** for better command line experience
4. **Close unnecessary applications** when processing large datasets

---

**Need Help?** Check the main [DEPLOYMENT.md](../DEPLOYMENT.md) for more detailed instructions.