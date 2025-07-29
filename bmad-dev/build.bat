@echo off
echo Building BMAD Multi-Pool KawPow Library...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

REM Build the project
echo Building project...
cmake --build . --config Release

REM Copy KawPow header from xmrig-cuda
echo Copying KawPow headers...
if not exist ..\include\KawPow mkdir ..\include\KawPow
if not exist ..\include\KawPow\raven mkdir ..\include\KawPow\raven
copy ..\..\xmrig-cuda\src\KawPow\raven\KawPow.h ..\include\KawPow\raven\
copy ..\..\xmrig-cuda\src\KawPow\raven\KawPow_dag.h ..\include\KawPow\raven\

echo Build complete!
echo Library location: build\lib\Release\bmad_kawpow_multi.dll
echo Headers location: build\include\bmad\

cd ..