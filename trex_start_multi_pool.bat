@echo off
:start
echo Starting T-Rex for Multi-Pool Mining
echo ===================================

REM Make sure the multi_pool_proxy.py is running first!
echo Make sure python multi_pool_proxy.py is running first!
echo.

REM T-Rex command to connect to your multi-pool proxy
t-rex.exe -a kawpow -o stratum+tcp://localhost:4444 -u RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU -w multi-rig --no-watchdog

echo.
echo Mining stopped. Press any key to restart...
pause
goto :start 