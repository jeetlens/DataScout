# 🔧 DataScout Windows Troubleshooting Guide

## Backend Issues

### Issue 1: WeasyPrint Import Error
**Error:** `OSError: cannot load library 'gobject-2.0-0'`

**✅ Solution:** 
1. The system now uses Windows-compatible PDF generation
2. Make sure ReportLab is installed:
   ```cmd
   pip install reportlab
   ```
3. If still failing, skip PDF generation temporarily by commenting out PDF imports

### Issue 2: Python Module Not Found
**Error:** `ModuleNotFoundError: No module named 'pandas'`

**✅ Solution:**
```cmd`
# Navigate to backend directory
cd backend

# Activate virtual environment
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Or install core packages manually
pip install fastapi uvicorn pandas numpy matplotlib plotly reportlab
```

### Issue 3: Virtual Environment Issues
**Error:** `'venv\Scripts\activate' is not recognized`

**✅ Solution:**
```cmd
# Create new virtual environment
python -m venv venv

# Use full path to activate
C:\Users\jeetr\Desktop\Project\DataScout\backend\venv\Scripts\activate.bat

# Or use PowerShell
venv\Scripts\Activate.ps1
```

### Issue 4: Python Not Found
**Error:** `'python' is not recognized`

**✅ Solution:**
1. Install Python 3.8+ from [python.org](https://python.org)
2. Make sure "Add to PATH" is checked during installation
3. Try using `py` instead of `python`:
   ```cmd
   py -m venv venv
   py main.py
   ```

## Frontend Issues

### Issue 1: Vite Command Not Found
**Error:** `'vite' is not recognized as an internal or external command`

**✅ Solution:**
```cmd
# Navigate to frontend directory
cd frontend

# Run setup script
setup_frontend_windows.bat

# Or install manually
npm install
npm run dev
```

### Issue 2: Node.js Not Found
**Error:** `'node' is not recognized`

**✅ Solution:**
1. Install Node.js 18+ from [nodejs.org](https://nodejs.org)
2. Restart Command Prompt after installation
3. Verify installation:
   ```cmd
   node --version
   npm --version
   ```

### Issue 3: npm Install Fails
**Error:** `npm ERR!` during installation

**✅ Solution:**
```cmd
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rmdir /s /q node_modules
del package-lock.json

# Try different registry
npm install --registry https://registry.npmjs.org/

# Or use yarn instead
npm install -g yarn
yarn install
```

### Issue 4: Port Already in Use
**Error:** `Port 5173 is already in use`

**✅ Solution:**
```cmd
# Find process using port 5173
netstat -ano | findstr :5173

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use different port
npm run dev -- --port 3000
```

## General Windows Issues

### Issue 1: Permission Denied
**✅ Solution:**
1. Run Command Prompt as Administrator
2. Or change folder permissions
3. Install in user directory:
   ```cmd
   pip install --user -r requirements.txt
   ```

### Issue 2: Antivirus Blocking
**✅ Solution:**
1. Add project folder to antivirus exclusions
2. Temporarily disable real-time protection
3. Use Windows Defender exclusions

### Issue 3: Long Path Names
**✅ Solution:**
1. Enable long path support in Windows
2. Move project to shorter path like `C:\DataScout`
3. Use PowerShell instead of Command Prompt

## Quick Start Commands

### Backend (Windows)
```cmd
cd backend
setup_windows.bat
python main.py
```

### Frontend (Windows)
```cmd
cd frontend
setup_frontend_windows.bat
start_frontend.bat
```

### Full Application
```cmd
# Terminal 1 - Backend
cd backend
venv\Scripts\activate
python main.py

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

## Environment Setup Verification

### Check Backend
```cmd
# Test Python imports
python -c "import fastapi, pandas; print('✅ Backend OK')"

# Test server health
curl http://localhost:8000/health
```

### Check Frontend
```cmd
# Test Node.js
node --version
npm --version

# Test Vite
npx vite --version

# Access application
# http://localhost:5173
```

## Alternative Solutions

### If All Else Fails - Docker Alternative
Since you're on Windows, you could also try:
1. Install Docker Desktop for Windows
2. Use the provided docker-compose.yml:
   ```cmd
   docker-compose up -d
   ```

### Minimal Setup
If you just want to test the backend:
```cmd
# Install only core packages
pip install fastapi uvicorn pandas

# Start with minimal server
python -c "
from fastapi import FastAPI
app = FastAPI()

@app.get('/')
def read_root():
    return {'status': 'DataScout Backend Running'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
"
```

## Contact & Support

If you're still having issues:
1. Check the error logs carefully
2. Make sure you're in the correct directory
3. Try the setup scripts provided
4. Ensure all prerequisites are installed

## Next Steps After Setup

Once both servers are running:
1. **Backend**: http://localhost:8000
2. **Frontend**: http://localhost:5173  
3. **API Docs**: http://localhost:8000/docs

Test the integration by uploading a CSV file and running analysis!