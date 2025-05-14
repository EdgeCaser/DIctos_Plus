@echo off

echo ===================================
echo DictosPlus Setup
echo ===================================
echo.

REM Check if Python 3.11 is available
py -3.11 --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.11 is required
    echo Please install Python 3.11 from python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Clean up any existing virtual environment
if exist .venv (
    echo Removing existing virtual environment...
    rmdir /s /q .venv
)

echo Creating fresh virtual environment...
py -3.11 -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo Installing dependencies...
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)

echo Installing NumPy 1.26.4...
pip install "numpy==1.26.4"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install NumPy
    pause
    exit /b 1
)

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)

echo Installing other dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install other dependencies
    pause
    exit /b 1
)

echo Starting DictosPlus...
python DictosPlus.py
pause
