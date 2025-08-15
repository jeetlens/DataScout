#!/bin/bash
# DataScout Backend Setup Script for Linux/macOS
# This script sets up the Python environment and installs dependencies

echo "Starting DataScout Backend Setup..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "Python found. Checking version..."
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3.8+ is required"
    python3 --version
    exit 1
fi

echo "Python version OK."
echo

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created successfully."
echo

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment activated."
echo

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to upgrade pip, continuing anyway..."
fi

echo

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    echo
    echo "Try installing manually:"
    echo "  pip install fastapi uvicorn pandas numpy matplotlib"
    exit 1
fi

echo
echo "✅ Setup completed successfully!"
echo
echo "To start the backend server:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run server: python main.py"
echo
echo "To deactivate virtual environment later: deactivate"
echo