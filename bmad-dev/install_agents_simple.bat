@echo off
echo Installing BMAD Multi-Pool KawPow Agents (Simple Setup)...
echo.

REM Create build directory structure
if not exist "build" mkdir build
if not exist "build\lib" mkdir build\lib
if not exist "build\lib\Release" mkdir build\lib\Release
if not exist "build\include" mkdir build\include
if not exist "build\include\bmad" mkdir build\include\bmad

REM Copy headers
echo Copying BMAD headers...
copy "include\*.h" "build\include\bmad\" >nul
if exist "include\KawPow" (
    xcopy "include\KawPow\*" "build\include\bmad\KawPow\" /E /Y >nul
)

REM Create placeholder library for testing
echo Creating placeholder BMAD library...
echo // Placeholder BMAD library for testing > build\lib\Release\bmad_kawpow_multi.dll
echo // This is a placeholder file. In production, this would be the actual CUDA library.

REM Create agent configuration
echo Creating BMAD agent configuration...
(
echo {
echo     "bmad_agents": {
echo         "kawpow_multi": {
echo             "enabled": true,
echo             "max_pools": 10,
echo             "batch_size": 1024,
echo             "memory_alignment": 4096,
echo             "use_pinned_memory": true,
echo             "enable_profiling": false
echo         }
echo     },
echo     "cuda_config": {
echo         "device_id": 0,
echo         "blocks": 8192,
echo         "threads": 32,
echo         "intensity": 262144
echo     },
echo     "pool_agents": [
echo         {
echo             "name": "agent_0",
echo             "pool_id": 0,
echo             "enabled": true,
echo             "priority": 1
echo         },
echo         {
echo             "name": "agent_1", 
echo             "pool_id": 1,
echo             "enabled": true,
echo             "priority": 2
echo         },
echo         {
echo             "name": "agent_2",
echo             "pool_id": 2,
echo             "enabled": true,
echo             "priority": 3
echo         },
echo         {
echo             "name": "agent_3",
echo             "pool_id": 3,
echo             "enabled": true,
echo             "priority": 4
echo         },
echo         {
echo             "name": "agent_4",
echo             "pool_id": 4,
echo             "enabled": true,
echo             "priority": 5
echo         }
echo     ]
echo }
) > bmad_agents_config.json

echo.
echo BMAD Agents Installation Complete!
echo.
echo Files created:
echo - build\lib\Release\bmad_kawpow_multi.dll (placeholder)
echo - build\include\bmad\ (headers)
echo - bmad_agents_config.json
echo.
echo Next steps:
echo 1. Copy bmad_kawpow_multi.dll to your XMRig directory
echo 2. Update your XMRig config to use BMAD agents
echo 3. Test multi-pool mining with BMAD
echo.
echo Note: This is a placeholder installation. For full functionality,
echo you need to build the actual CUDA library with proper CUDA tools.
echo.
pause