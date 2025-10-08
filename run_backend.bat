@echo off
CLS

:START_SERVER
echo.
echo ===================================================
echo  Backend server starting... (Press Ctrl+C to stop)
echo ===================================================
echo.

rem --- 서버를 실행합니다. Ctrl+C로 종료하면 이 줄 아래로 넘어갑니다. ---
c:\code\.venv\Scripts\python.exe c:\code\customMydataService\backend\app.py

echo.
echo Server stopped.

rem --- 재시작 또는 종료를 선택합니다. ---
CHOICE /C YN /M "Do you want to restart the server? (Y/N)"

rem --- 선택 결과에 따라 분기합니다. ---
IF ERRORLEVEL 2 GOTO :EOF
IF ERRORLEVEL 1 GOTO :START_SERVER

:EOF
echo Exiting script.
pause