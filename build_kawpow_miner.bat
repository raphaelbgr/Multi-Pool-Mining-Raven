@echo off
echo Building KAWPOW Multi-Pool Miner...
echo.

echo Compiling kawpow_multi_target.cu...
nvcc kawpow_multi_target.cu -o kawpow_multi_target.exe -Xcompiler="/EHsc" -lcuda --ptxas-options=-v

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Successfully built kawpow_multi_target.exe
    echo.
    echo 🚀 Ready to mine Ravencoin with KAWPOW!
    echo.
    echo To start mining:
    echo   python kawpow_multi_pool_miner.py
    echo.
) else (
    echo.
    echo ❌ Build failed!
    echo.
) 