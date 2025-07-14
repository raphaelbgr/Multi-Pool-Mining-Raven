@echo off
echo ===============================================
echo OPTIMIZED MULTI-POOL MINING SYSTEM
echo ===============================================
echo.
echo This system tests each nonce against ALL pool targets simultaneously!
echo Much more efficient than testing separate nonces for each pool.
echo.

echo [1/3] Building optimized miner...
call build_multi_target.bat

echo.
echo [2/3] Getting fresh jobs from all pools...
python get_jobs.py

echo.
echo [3/3] Starting optimized multi-pool mining...
echo Press Ctrl+C to stop mining
echo.

python auto_miner_optimized.py

pause 