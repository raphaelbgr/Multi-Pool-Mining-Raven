@echo off
echo Iniciando minerador na RTX 3080 Ti...
set CUDA_VISIBLE_DEVICES=0
python auto_miner.py
pause
