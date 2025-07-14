@echo off
echo ========================================
echo    Multi-Pool Miner - RTX 3080 Ti LHR
echo ========================================
echo.

echo Configurando para RTX 3080 Ti LHR...
set CUDA_VISIBLE_DEVICES=0
set CUDA_LAUNCH_BLOCKING=0
set CUDA_CACHE_DISABLE=0
set CUDA_FORCE_PTX_JIT=0
set CUDA_MEMORY_POOL_SIZE=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID

echo.
echo IMPORTANTE: GPU LHR tem limitações de hashrate!
echo Performance esperada: 1.5-2.0 GH/s
echo.
pause

echo.
echo Iniciando minerador LHR otimizado...
echo.

python auto_miner.py

echo.
echo Minerador parado.
pause
