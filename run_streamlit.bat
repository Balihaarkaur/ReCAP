@echo off
REM Detached runner for Streamlit — writes logs to streamlit.log
SET VENV_PY=C:\Users\sarab\OneDrive\Desktop\Moveewise\.venv\Scripts\python.exe
CD /D %~dp0
echo Starting Streamlit with venv: %VENV_PY%
"%VENV_PY%" -u -m streamlit run app.py --server.port 8501 --server.headless true --logger.level debug > streamlit.log 2>&1
echo Streamlit exited. Press any key to close.
pause