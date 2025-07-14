@echo off
echo ========================================
echo    Multi-Pool Miner - RTX 3080 Ti OTIMIZADO
echo ========================================
echo.

echo Configurando ambiente otimizado...
set CUDA_VISIBLE_DEVICES=0
set CUDA_LAUNCH_BLOCKING=0
set CUDA_CACHE_DISABLE=0
set CUDA_FORCE_PTX_JIT=0

echo.
echo IMPORTANTE: Pare o BitCrack para melhor performance!
echo.
pause

echo.
echo Iniciando minerador otimizado...
echo.

python auto_miner.py

echo.
echo Minerador parado.
pause
