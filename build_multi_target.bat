@echo off
echo Building Optimized Multi-Pool Miner...
set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set PATH=%CUDA_PATH%\bin;%PATH%

echo Compiling miner_multi_target.cu...
nvcc miner_multi_target.cu -o miner_multi_target.exe -Xcompiler="/EHsc" -lcuda --ptxas-options=-v

if %ERRORLEVEL% EQU 0 (
    echo ✅ Successfully built miner_multi_target.exe
    echo 🎯 This miner tests each nonce against ALL pool targets simultaneously!
) else (
    echo ❌ Build failed!
)

pause 