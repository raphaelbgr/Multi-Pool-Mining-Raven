@echo off
echo Building X16R CUDA Miner for Ravencoin...
echo ========================================

REM Check if nvcc is available
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found. Please install CUDA Toolkit.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

echo Compiling X16R miner...
nvcc -O3 -arch=sm_60 -o miner_x16r.exe x16r_cuda_miner.cu

if errorlevel 1 (
    echo ERROR: Compilation failed!
    pause
    exit /b 1
)

echo.
echo SUCCESS: X16R miner compiled successfully!
echo.
echo Usage:
echo   miner_x16r.exe <start_nonce> [algorithm]
echo.
echo Examples:
echo   miner_x16r.exe 0          # Use X16R algorithm (default)
echo   miner_x16r.exe 0 sha256   # Use SHA-256 algorithm (legacy)
echo.
echo The X16R miner includes:
echo - Official X16R algorithm implementation
echo - 16 different hash algorithms (BLAKE, BMW, GROESTL, etc.)
echo - Proper little-endian nonce handling
echo - Multi-pool support
echo - Backward compatibility with SHA-256
echo.

REM Test compilation
echo Testing compilation...
echo 123456 | miner_x16r.exe 0 >nul 2>&1
if errorlevel 1 (
    echo WARNING: Test run failed, but compilation succeeded.
    echo This may be normal if headers.bin is missing.
) else (
    echo Test run successful!
)

echo.
echo Ready to mine Ravencoin with X16R algorithm!
echo Run: python auto_miner_optimized.py
pause 