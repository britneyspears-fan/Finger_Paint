@echo off
cd /d "%~dp0"
echo ========================================================
echo Iniciando servidor local en http://localhost:8000 ...
echo ========================================================
start http://localhost:8000
python -m http.server 8000
if %errorlevel% neq 0 (
    echo Probando con lanzador py...
    py -m http.server 8000
)
pause
