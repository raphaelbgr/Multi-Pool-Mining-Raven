@echo off
echo ========================================
echo BMAD Test Suite Runner
echo ========================================
echo.

cd /d "%~dp0build"

echo Building all test executables...
cmake --build . --config Release
if %ERRORLEVEL% neq 0 (
    echo ❌ Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Running Unit Tests
echo ========================================
echo.
Release\test_unit_components.exe
set UNIT_RESULT=%ERRORLEVEL%

echo.
echo ========================================
echo Running Integration Tests
echo ========================================
echo.
Release\test_bmad_integration.exe
set INTEGRATION_RESULT=%ERRORLEVEL%

echo.
echo ========================================
echo Test Summary
echo ========================================
echo.

if %UNIT_RESULT% equ 0 (
    echo ✅ Unit Tests: PASSED
) else (
    echo ❌ Unit Tests: FAILED
)

if %INTEGRATION_RESULT% equ 0 (
    echo ✅ Integration Tests: PASSED
) else (
    echo ❌ Integration Tests: FAILED
)

if %UNIT_RESULT% equ 0 if %INTEGRATION_RESULT% equ 0 (
    echo.
    echo 🎉 All tests passed! BMAD integration is working correctly.
    echo.
    echo Next steps:
    echo 1. Test with real Ravencoin pools
    echo 2. Integrate with XMRig
    echo 3. Optimize performance
    echo 4. Add more error handling
) else (
    echo.
    echo ⚠️ Some tests failed. Please review the output above.
    echo.
    echo Troubleshooting:
    echo 1. Check that all dependencies are installed
    echo 2. Verify CUDA installation
    echo 3. Review error messages in test output
)

echo.
pause 