@echo off
cd /d %~dp0
python -m venv venv
call venv\Scripts\activate
pip install -r backend\requirements.txt
python backend\app.py
