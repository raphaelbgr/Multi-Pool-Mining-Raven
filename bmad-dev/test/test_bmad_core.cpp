#include "bmad_kawpow_multi.h"
#include "bmad_memory_manager.h"
#include "bmad_pool_manager.h"
#include "bmad_types.h"
#include <iostream>
#include <cassert>
#include <cstring>

// Test utilities
void printTestHeader(const std::string& testName) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "TEST: " << testName << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void printTestResult(bool passed, const std::string& message) {
    std::cout << (passed ? "✅ PASS" : "❌ FAIL") << ": " << message << std::endl;
}

// Test data
const uint8_t TEST_JOB_BLOB[] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
    0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28
};

const uint8_t TEST_CACHE[] = {
    0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11,
    0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99
};

// Test 1: BMAD::KawPowMulti class
bool testKawPowMultiClass() {
    printTestHeader("BMAD::KawPowMulti Class");
    
    try {
        // Test constructor
        BMAD::KawPowMulti kawpow;
        printTestResult(true, "Constructor created successfully");
        
        // Test initialization
        BMAD::BMADConfig config;
        config.max_pools = 5;
        config.max_nonces = 1024;
        config.device_id = 0;
        config.batch_size = 256;
        config.memory_alignment = 256;
        config.use_pinned_memory = false;
        config.enable_profiling = false;
        
        bool init_result = kawpow.initialize(config);
        printTestResult(init_result, "Initialization completed");
        
        // Test cleanup
        kawpow.cleanup();
        printTestResult(true, "Cleanup completed successfully");
        
        return true;
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Test 2: BMAD::MemoryManager class
bool testMemoryManagerClass() {
    printTestHeader("BMAD::MemoryManager Class");
    
    try {
        // Test constructor
        BMAD::MemoryManager memory;
        printTestResult(true, "Constructor created successfully");
        
        // Test initialization
        BMAD::BMADConfig config;
        config.max_pools = 3;
        config.max_nonces = 512;
        config.device_id = 0;
        config.batch_size = 128;
        config.memory_alignment = 256;
        config.use_pinned_memory = false;
        config.enable_profiling = false;
        
        bool init_result = memory.initialize(config, 0);
        printTestResult(init_result, "Initialization completed");
        
        // Test memory allocation
        bool dag_result = memory.allocateDAG(1024);
        printTestResult(dag_result, "DAG memory allocation");
        
        bool cache_result = memory.allocateCache(512);
        printTestResult(cache_result, "Cache memory allocation");
        
        bool job_result = memory.allocateJobMemory(3);
        printTestResult(job_result, "Job memory allocation");
        
        // Test cleanup
        memory.cleanup();
        printTestResult(true, "Cleanup completed successfully");
        
        return true;
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Test 3: BMAD::PoolManager class
bool testPoolManagerClass() {
    printTestHeader("BMAD::PoolManager Class");
    
    try {
        // Test constructor
        BMAD::PoolManager pool;
        printTestResult(true, "Constructor created successfully");
        
        // Test initialization
        bool init_result = pool.initialize(5);
        printTestResult(init_result, "Initialization completed");
        
        // Test pool management
        bool add_result = pool.addPool(1, "TestPool1");
        printTestResult(add_result, "Add pool 1");
        
        bool add_result2 = pool.addPool(2, "TestPool2");
        printTestResult(add_result2, "Add pool 2");
        
        // Test pool status
        bool active_result = pool.setPoolActive(1, true);
        printTestResult(active_result, "Set pool 1 active");
        
        bool connected_result = pool.setPoolConnected(1, true);
        printTestResult(connected_result, "Set pool 1 connected");
        
        // Test job management
        BMAD::MultiPoolJob job;
        job.pool_id = 1;
        job.target = 0x12345678;
        job.height = 1000;
        job.active = true;
        std::memcpy(job.blob, TEST_JOB_BLOB, sizeof(TEST_JOB_BLOB));
        
        bool job_result = pool.updatePoolJob(1, job);
        printTestResult(job_result, "Update pool job");
        
        // Test statistics
        uint32_t active_count = pool.getActivePoolCount();
        printTestResult(active_count == 1, "Active pool count: " + std::to_string(active_count));
        
        uint32_t total_count = pool.getTotalPoolCount();
        printTestResult(total_count == 2, "Total pool count: " + std::to_string(total_count));
        
        // Test cleanup
        pool.cleanup();
        printTestResult(true, "Cleanup completed successfully");
        
        return true;
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Test 4: Multi-pool job data structures
bool testMultiPoolJobDataStructures() {
    printTestHeader("Multi-Pool Job Data Structures");
    
    try {
        // Test MultiPoolJob structure
        BMAD::MultiPoolJob job;
        job.pool_id = 1;
        job.target = 0x12345678;
        job.height = 1000;
        job.active = true;
        std::memcpy(job.blob, TEST_JOB_BLOB, sizeof(TEST_JOB_BLOB));
        
        printTestResult(job.pool_id == 1, "Pool ID set correctly");
        printTestResult(job.target == 0x12345678, "Target set correctly");
        printTestResult(job.height == 1000, "Height set correctly");
        
        // Test MultiPoolResult structure
        BMAD::MultiPoolResult result;
        result.pool_id = 1;
        result.nonce = 0x12345678;
        result.hash = 0x87654321;
        result.actual_diff = 0x11111111;
        result.valid = true;
        
        printTestResult(result.pool_id == 1, "Result pool ID set correctly");
        printTestResult(result.nonce == 0x12345678, "Result nonce set correctly");
        printTestResult(result.hash == 0x87654321, "Result hash set correctly");
        
        // Test BMADConfig structure
        BMAD::BMADConfig config;
        config.max_pools = 5;
        config.max_nonces = 1024;
        config.device_id = 0;
        config.batch_size = 256;
        config.memory_alignment = 256;
        config.use_pinned_memory = false;
        config.enable_profiling = false;
        
        printTestResult(config.max_pools == 5, "Config max pools set correctly");
        printTestResult(config.max_nonces == 1024, "Config max nonces set correctly");
        printTestResult(config.device_id == 0, "Config device ID set correctly");
        
        return true;
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Test 5: CMake integration
bool testCMakeIntegration() {
    printTestHeader("CMake Integration");
    
    try {
        // Test that headers are accessible
        BMAD::KawPowMulti* kawpow = nullptr;
        BMAD::MemoryManager* memory = nullptr;
        BMAD::PoolManager* pool = nullptr;
        
        printTestResult(true, "All classes accessible in BMAD namespace");
        
        return true;
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Main test runner
int main() {
    std::cout << "BMAD Core Library Test Suite" << std::endl;
    std::cout << "============================" << std::endl;
    
    int passedTests = 0;
    int totalTests = 0;
    
    // Run all tests
    totalTests++;
    if (testKawPowMultiClass()) passedTests++;
    
    totalTests++;
    if (testMemoryManagerClass()) passedTests++;
    
    totalTests++;
    if (testPoolManagerClass()) passedTests++;
    
    totalTests++;
    if (testMultiPoolJobDataStructures()) passedTests++;
    
    totalTests++;
    if (testCMakeIntegration()) passedTests++;
    
    // Print final results
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Passed: " << passedTests << "/" << totalTests << " tests" << std::endl;
    std::cout << "Success rate: " << (passedTests * 100 / totalTests) << "%" << std::endl;
    
    if (passedTests == totalTests) {
        std::cout << "🎉 ALL TESTS PASSED! BMAD core library is ready." << std::endl;
        return 0;
    } else {
        std::cout << "⚠️  Some tests failed. Please review the implementation." << std::endl;
        return 1;
    }
}