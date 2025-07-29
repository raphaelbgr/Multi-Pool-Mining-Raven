# Test framework for BMAD Multi-Pool KawPow Mining

# Enable testing
enable_testing()

# Test source files
set(TEST_SOURCES
    test_bmad_core.cpp
)

# Create test executable
add_executable(bmad_test ${TEST_SOURCES})

# Link with BMAD library
target_link_libraries(bmad_test bmad_kawpow_multi)

# Include directories
target_include_directories(bmad_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Set runtime path for DLL loading
set_target_properties(bmad_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add test to CTest
add_test(NAME BMAD_Core_Tests COMMAND bmad_test)

# Set test working directory
set_tests_properties(BMAD_Core_Tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)