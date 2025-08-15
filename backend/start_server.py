#!/usr/bin/env python3
"""
DataScout Backend Startup Script
Ensures proper environment setup before starting the server.
"""

import sys
import os
import subprocess

def main():
    # Get the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = backend_dir
    env['PYTHONUNBUFFERED'] = '1'
    
    # Find the virtual environment Python executable
    venv_python = None
    possible_paths = [
        os.path.join(backend_dir, '..', 'venv', 'Scripts', 'python.exe'),  # Windows
        os.path.join(backend_dir, '..', 'venv', 'bin', 'python'),         # Unix
        os.path.join(backend_dir, 'venv', 'Scripts', 'python.exe'),       # Windows (local venv)
        os.path.join(backend_dir, 'venv', 'bin', 'python'),               # Unix (local venv)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            venv_python = path
            break
    
    if not venv_python:
        print("Error: Could not find virtual environment Python executable")
        print("Please ensure the virtual environment is set up correctly")
        return 1
    
    print(f"Using Python: {venv_python}")
    print(f"Backend directory: {backend_dir}")
    print(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}")
    print("Starting DataScout backend server...")
    
    # Start the server
    try:
        subprocess.run([
            venv_python, 
            os.path.join(backend_dir, 'main.py')
        ], env=env, cwd=backend_dir)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return 0
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())