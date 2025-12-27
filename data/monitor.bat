@echo off
:loop
cls
echo ============================================================
echo ORBITAAL DOWNLOAD MONITOR
echo ============================================================
echo.
echo Current downloads:
echo.
for %%f in (C:\Users\kevin\livetrading\data\orbitaal\*.tar.gz) do (
    echo %%~nxf: %%~zf bytes
)
echo.
echo Active curl processes:
tasklist /FI "IMAGENAME eq curl.exe" 2>nul | find "curl"
echo.
echo Refresh in 10 seconds... (Ctrl+C to stop)
timeout /t 10 >nul
goto loop
