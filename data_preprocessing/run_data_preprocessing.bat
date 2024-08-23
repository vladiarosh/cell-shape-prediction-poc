@echo off
CALL %~dp0\venv\Scripts\activate
python %~dp0\run_data_preprocessing.py
cmd /k