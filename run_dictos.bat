@echo off
IF NOT EXIST venv (
    echo Creating virtual environment...
    py -3.11 -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
) ELSE (
    call venv\Scripts\activate
)

echo Starting DictosPlus...
python DictosPlus.py
pause
