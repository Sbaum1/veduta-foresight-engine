@echo off
title VEDUTA Foresight X
cd /d "%~dp0"
echo.
echo  VEDUTA ^| Foresight X
echo  38 Models ^| M3 MASE 0.6847 ^| #1 Modern Published
echo  -----------------------------------------------
echo  Starting on http://localhost:8501
echo.
start "" "http://localhost:8501"
"V:\.venv\Scripts\streamlit.exe" run app.py --server.port 8501
pause