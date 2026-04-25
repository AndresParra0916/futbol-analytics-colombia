@echo off
cd /d "C:\Users\USER\Documents\futbol-analytics-colombia"
call .venv\Scripts\activate
streamlit run app.py --server.headless true
pause