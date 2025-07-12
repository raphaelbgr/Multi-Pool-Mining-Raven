@echo off
set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set PATH=%CUDA_PATH%\bin;%PATH%
nvcc miner.cu -o miner.exe -Xcompiler="/EHsc" -lcuda -lssl -lcrypto
pause