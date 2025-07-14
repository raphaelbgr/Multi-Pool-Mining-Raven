@echo off
echo Building Proper X16R CUDA Miner for Ravencoin...
echo ================================================
echo Based on official Ravencoin implementation
echo.

REM Check if nvcc is available
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found. Please install CUDA Toolkit.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

echo Compiling Proper X16R miner...
nvcc -O3 -arch=sm_60 -o miner_x16r_proper.exe x16r_cuda_miner_proper.cu

if errorlevel 1 (
    echo ERROR: Compilation failed!
    pause
    exit /b 1
)

echo.
echo SUCCESS: Proper X16R miner compiled successfully!
echo.
echo Usage:
echo   miner_x16r_proper.exe <start_nonce>
echo.
echo Examples:
echo   miner_x16r_proper.exe 0          # Start from nonce 0
echo   miner_x16r_proper.exe 1000000    # Start from nonce 1000000
echo.
echo The Proper X16R miner includes:
echo - Official X16R algorithm implementation
echo - 16 different hash algorithms (BLAKE, BMW, GROESTL, etc.)
echo - Proper hash selection based on previous block hash
echo - Multi-pool support
echo - Based on official Ravencoin source code
echo.

REM Test compilation
echo Testing compilation...
echo 123456 | miner_x16r_proper.exe 0 >nul 2>&1
if errorlevel 1 (
    echo WARNING: Test run failed, but compilation succeeded.
    echo This may be normal if headers.bin is missing.
) else (
    echo Test run successful!
)

echo.
echo Ready to mine Ravencoin with proper X16R algorithm!
echo Run: python auto_miner_x16r.py
pause 