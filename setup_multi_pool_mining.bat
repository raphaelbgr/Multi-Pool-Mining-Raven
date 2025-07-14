@echo off
echo Multi-Pool Ravencoin Mining Setup
echo ===================================
echo.

echo Step 1: Download T-Rex Miner
echo ----------------------------
echo Please download T-Rex from: https://github.com/trexminer/T-Rex/releases
echo - Download the latest t-rex-X.X.X-win-cuda11.8.zip
echo - Extract it to a folder (e.g., C:\T-Rex\)
echo.

echo Step 2: Your Multi-Pool Configuration
echo ------------------------------------
echo Your pools configured in config.json:
echo - 2Miners (rvn.2miners.com:6060)
echo - Ravenminer (stratum.ravenminer.com:3838)  
echo - WoolyPooly (pool.br.woolypooly.com:55555)
echo - HeroMiners (br.ravencoin.herominers.com:1140)
echo - Nanopool (rvn-us-east1.nanopool.org:10400)
echo.

echo Step 3: Start the Multi-Pool Proxy
echo ----------------------------------
echo 1. Open Command Prompt in this folder
echo 2. Run: python multi_pool_proxy.py
echo 3. Leave this running - it will distribute shares to all pools
echo.

echo Step 4: Start T-Rex Miner
echo -------------------------
echo In T-Rex folder, create start_mining.bat with this content:
echo.
echo t-rex.exe -a kawpow -o stratum+tcp://localhost:4444 -u RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU -w multi-rig
echo pause
echo.

echo Step 5: Run Both
echo ----------------
echo 1. Start multi_pool_proxy.py (this script)
echo 2. Start start_mining.bat (T-Rex connecting to proxy)
echo 3. Watch as shares get distributed to ALL pools!
echo.

pause 