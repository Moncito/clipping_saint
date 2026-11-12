@echo off
rem Double-click to open the Clipper app.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" clipper_app.py
