cmake_minimum_required(VERSION 3.16)
project(BMAD_MultiPool_KawPow CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files (C++ only for now)
set(SOURCES
    src/bmad_kawpow_host.cpp
    src/bmad_pool_manager.cpp
    src/bmad_memory_manager.cpp
    src/bmad_kawpow_algorithm.cpp
    src/bmad_kawpow_optimized.cpp
    src/bmad_gpu_memory_manager.cpp
    src/bmad_performance_tester.cpp
    src/bmad_xmrig_bridge.cpp
    src/bmad_share_converter.cpp
    src/bmad_pool_connector.cpp
)

# Header files
set(HEADERS
    include/bmad_kawpow_multi.h
    include/bmad_memory_manager.h
    include/bmad_pool_manager.h
    include/bmad_types.h
    include/bmad_agent_manager.h
    include/bmad_kawpow_algorithm.h
    include/bmad_kawpow_optimized.h
    include/bmad_gpu_memory_manager.h
    include/bmad_performance_tester.h
    include/bmad_xmrig_bridge.h
    include/bmad_share_converter.h
    include/bmad_pool_connector.h
)

# Create shared library
add_library(bmad_kawpow_multi SHARED ${SOURCES})

# Create static library for linking
add_library(bmad_kawpow_multi_static STATIC ${SOURCES})
set_target_properties(bmad_kawpow_multi_static PROPERTIES OUTPUT_NAME bmad_kawpow_multi)

# Set output directory
set_target_properties(bmad_kawpow_multi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(bmad_kawpow_multi_static PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Include directories for the library
target_include_directories(bmad_kawpow_multi PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(bmad_kawpow_multi_static PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Create test executable
add_executable(test_multi_pool test_simple_multi.cpp)
target_link_libraries(test_multi_pool bmad_kawpow_multi_static)
target_include_directories(test_multi_pool PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create integration test executable
add_executable(test_bmad_integration test_bmad_integration.cpp)
target_link_libraries(test_bmad_integration bmad_kawpow_multi_static)
target_include_directories(test_bmad_integration PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create unit test executable
add_executable(test_unit_components test_unit_components.cpp)
target_link_libraries(test_unit_components bmad_kawpow_multi_static)
target_include_directories(test_unit_components PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create real pool test executable
add_executable(test_real_ravencoin_pools test_real_ravencoin_pools.cpp)
target_link_libraries(test_real_ravencoin_pools bmad_kawpow_multi_static)
target_include_directories(test_real_ravencoin_pools PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create performance test executable
add_executable(test_performance test_performance_comparison.cpp)
target_link_libraries(test_performance bmad_kawpow_multi_static)
target_include_directories(test_performance PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create different jobs demonstration executable
add_executable(test_different_jobs test_different_jobs_demo.cpp)
target_link_libraries(test_different_jobs bmad_kawpow_multi_static)
target_include_directories(test_different_jobs PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create real pool test executable
add_executable(test_real_pools src/bmad_real_pool_test.cpp)
target_link_libraries(test_real_pools bmad_kawpow_multi_static)
target_include_directories(test_real_pools PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create optimized kernel test executable
add_executable(test_optimized_kernel test_optimized_kernel.cpp)
target_link_libraries(test_optimized_kernel bmad_kawpow_multi_static)
target_include_directories(test_optimized_kernel PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create comprehensive test suite executable
add_executable(test_comprehensive_suite test_comprehensive_suite.cpp)
target_link_libraries(test_comprehensive_suite bmad_kawpow_multi_static)
target_include_directories(test_comprehensive_suite PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create quick check test executable
add_executable(test_quick_check test_quick_check.cpp)
target_link_libraries(test_quick_check bmad_kawpow_multi_static)
target_include_directories(test_quick_check PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create GPU memory test executable
add_executable(test_gpu_memory test_gpu_memory.cpp)
target_link_libraries(test_gpu_memory bmad_kawpow_multi_static)
target_include_directories(test_gpu_memory PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create performance tester executable
add_executable(test_performance_tester test_performance_tester.cpp)
target_link_libraries(test_performance_tester bmad_kawpow_multi_static)
target_include_directories(test_performance_tester PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create real mining test executable
add_executable(test_real_mining src/bmad_real_mining_test.cpp)
target_link_libraries(test_real_mining bmad_kawpow_multi_static)
target_include_directories(test_real_mining PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create real mining with XMRig integration executable
add_executable(test_real_mining_xmrig test_real_mining_with_xmrig.cpp)
target_link_libraries(test_real_mining_xmrig bmad_kawpow_multi_static)
target_include_directories(test_real_mining_xmrig PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create real mining with actual pools executable
add_executable(test_real_mining_actual_pools test_real_mining_actual_pools.cpp)
target_link_libraries(test_real_mining_actual_pools bmad_kawpow_multi_static)
target_include_directories(test_real_mining_actual_pools PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Create real mining monitoring executable
add_executable(test_real_mining_monitoring test_real_mining_monitoring.cpp)
target_link_libraries(test_real_mining_monitoring bmad_kawpow_multi_static)
target_include_directories(test_real_mining_monitoring PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Install rules
install(TARGETS bmad_kawpow_multi bmad_kawpow_multi_static
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)

install(FILES ${HEADERS} DESTINATION include/bmad)

# Add test subdirectory
add_subdirectory(test)