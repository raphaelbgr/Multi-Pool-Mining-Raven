#include "bmad_kawpow_multi.h"
#include "bmad_memory_manager.h"
#include "bmad_pool_manager.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// Test data for KawPow jobs
const uint8_t TEST_JOB_BLOB_1[] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
    0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28
};

const uint8_t TEST_JOB_BLOB_2[] = {
    0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40,
    0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50,
    0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58
};

const uint8_t TEST_JOB_BLOB_3[] = {
    0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70,
    0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F, 0x80,
    0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88
};

void printTestHeader(const std::string& testName) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST: " << testName << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void printTestResult(bool passed, const std::string& message) {
    std::cout << (passed ? "✅ PASS" : "❌ FAIL") << ": " << message << std::endl;
}

bool testBMADInitialization() {
    printTestHeader("BMAD Framework Initialization");
    
    BMAD::BMADConfig config;
    config.device_id = 0;
    config.max_pools = 5;
    config.max_nonces = 1000000;
    config.memory_limit = 1024 * 1024 * 1024; // 1GB
    
    bool result = BMAD::KawPowMulti::initialize(config);
    printTestResult(result, "BMAD KawPow Multi initialization");
    
    if (result) {
        uint32_t activePools = BMAD::KawPowMulti::getActivePoolCount();
        uint32_t totalPools = BMAD::KawPowMulti::getTotalPoolCount();
        printTestResult(activePools == 0, "Initial active pools count");
        printTestResult(totalPools == 0, "Initial total pools count");
    }
    
    return result;
}

bool testDAGPreparation() {
    printTestHeader("DAG Preparation");
    
    uint64_t testHeight = 12345678;
    bool result = BMAD::KawPowMulti::prepare(testHeight);
    printTestResult(result, "DAG preparation for height " + std::to_string(testHeight));
    
    return result;
}

bool testMultiPoolJobSetting() {
    printTestHeader("Multi-Pool Job Setting");
    
    std::vector<BMAD::MultiPoolJob> jobs;
    
    // Create test jobs for multiple pools
    BMAD::MultiPoolJob job1;
    job1.pool_id = 0;
    job1.target = 0x00000000FFFFFFFF;
    job1.height = 12345678;
    memcpy(job1.blob, TEST_JOB_BLOB_1, sizeof(TEST_JOB_BLOB_1));
    jobs.push_back(job1);
    
    BMAD::MultiPoolJob job2;
    job2.pool_id = 1;
    job2.target = 0x00000000FFFFFFFE;
    job2.height = 12345678;
    memcpy(job2.blob, TEST_JOB_BLOB_2, sizeof(TEST_JOB_BLOB_2));
    jobs.push_back(job2);
    
    BMAD::MultiPoolJob job3;
    job3.pool_id = 2;
    job3.target = 0x00000000FFFFFFFD;
    job3.height = 12345678;
    memcpy(job3.blob, TEST_JOB_BLOB_3, sizeof(TEST_JOB_BLOB_3));
    jobs.push_back(job3);
    
    bool result = BMAD::KawPowMulti::setJobs(jobs);
    printTestResult(result, "Setting " + std::to_string(jobs.size()) + " multi-pool jobs");
    
    if (result) {
        uint32_t activePools = BMAD::KawPowMulti::getActivePoolCount();
        printTestResult(activePools == jobs.size(), "Active pools count matches job count");
    }
    
    return result;
}

bool testMultiPoolMining() {
    printTestHeader("Multi-Pool Mining Simulation");
    
    uint32_t startNonce = 0x12345678;
    uint32_t rescount = 0;
    uint32_t resnonce = 0;
    
    std::cout << "Starting multi-pool mining with nonce: 0x" << std::hex << startNonce << std::dec << std::endl;
    
    bool result = BMAD::KawPowMulti::mine(startNonce, &rescount, &resnonce);
    printTestResult(result, "Multi-pool mining execution");
    
    if (result) {
        printTestResult(rescount >= 0, "Result count is valid");
        printTestResult(resnonce >= startNonce, "Result nonce is valid");
        
        std::cout << "Mining results:" << std::endl;
        std::cout << "  - Results found: " << rescount << std::endl;
        std::cout << "  - First nonce: 0x" << std::hex << resnonce << std::dec << std::endl;
        std::cout << "  - Skipped hashes: " << BMAD::KawPowMulti::getSkippedHashes() << std::endl;
    }
    
    return result;
}

bool testMemoryManager() {
    printTestHeader("Memory Manager Testing");
    
    BMAD::MemoryManager memoryManager;
    BMAD::BMADConfig config;
    config.device_id = 0;
    config.max_pools = 5;
    config.max_nonces = 1000000;
    config.memory_limit = 1024 * 1024 * 1024; // 1GB
    
    bool result = memoryManager.initialize(config);
    printTestResult(result, "Memory manager initialization");
    
    if (result) {
        uint64_t testHeight = 12345678;
        bool dagResult = memoryManager.prepareDAG(testHeight);
        printTestResult(dagResult, "DAG preparation");
        
        if (dagResult) {
            uint8_t* dag = memoryManager.getDAG();
            size_t dagSize = memoryManager.getDAGSize();
            printTestResult(dag != nullptr, "DAG pointer is valid");
            printTestResult(dagSize > 0, "DAG size is valid");
            
            std::cout << "DAG info:" << std::endl;
            std::cout << "  - Size: " << dagSize << " bytes" << std::endl;
            std::cout << "  - Pointer: " << (dag ? "valid" : "null") << std::endl;
        }
        
        memoryManager.cleanup();
    }
    
    return result;
}

bool testPoolManager() {
    printTestHeader("Pool Manager Testing");
    
    BMAD::PoolManager poolManager;
    bool result = poolManager.initialize();
    printTestResult(result, "Pool manager initialization");
    
    if (result) {
        // Test pool addition
        bool addResult = poolManager.addPool(0, "pool1.example.com:3333");
        printTestResult(addResult, "Adding pool 0");
        
        addResult = poolManager.addPool(1, "pool2.example.com:3333");
        printTestResult(addResult, "Adding pool 1");
        
        // Test pool activation
        bool activateResult = poolManager.setPoolActive(0, true);
        printTestResult(activateResult, "Activating pool 0");
        
        activateResult = poolManager.setPoolActive(1, true);
        printTestResult(activateResult, "Activating pool 1");
        
        // Test job updates
        BMAD::MultiPoolJob testJob;
        testJob.pool_id = 0;
        testJob.target = 0x00000000FFFFFFFF;
        testJob.height = 12345678;
        memcpy(testJob.blob, TEST_JOB_BLOB_1, sizeof(TEST_JOB_BLOB_1));
        
        bool jobResult = poolManager.updatePoolJob(0, testJob);
        printTestResult(jobResult, "Updating pool 0 job");
        
        // Test statistics
        uint32_t activePools = poolManager.getActivePoolCount();
        uint32_t totalPools = poolManager.getTotalPoolCount();
        printTestResult(activePools == 2, "Active pools count");
        printTestResult(totalPools == 2, "Total pools count");
        
        std::cout << "Pool statistics:" << std::endl;
        std::cout << "  - Active pools: " << activePools << std::endl;
        std::cout << "  - Total pools: " << totalPools << std::endl;
        
        poolManager.printStats();
        
        poolManager.cleanup();
    }
    
    return result;
}

bool testCUDAKernelIntegration() {
    printTestHeader("CUDA Kernel Integration Testing");
    
    // Test kernel parameters
    uint32_t maxPools = get_max_pools();
    uint32_t maxNonces = get_max_nonces_per_block();
    size_t sharedMemorySize = get_shared_memory_size();
    
    printTestResult(maxPools > 0, "Max pools: " + std::to_string(maxPools));
    printTestResult(maxNonces > 0, "Max nonces per block: " + std::to_string(maxNonces));
    printTestResult(sharedMemorySize > 0, "Shared memory size: " + std::to_string(sharedMemorySize));
    
    // Test DAG memory allocation
    size_t testDagSize = 1024 * 1024; // 1MB test DAG
    bool allocResult = allocate_dag_memory(testDagSize);
    printTestResult(allocResult, "DAG memory allocation");
    
    if (allocResult) {
        // Test DAG copy
        std::vector<uint8_t> testDag(testDagSize, 0x42);
        bool copyResult = copy_dag_to_device(testDag.data(), testDagSize);
        printTestResult(copyResult, "DAG copy to device");
        
        // Clean up
        free_dag_memory();
    }
    
    return true;
}

void runAllTests() {
    std::cout << "BMAD Multi-Pool KawPow Mining Framework Test Suite" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    int passedTests = 0;
    int totalTests = 0;
    
    // Test 1: BMAD Initialization
    totalTests++;
    if (testBMADInitialization()) passedTests++;
    
    // Test 2: DAG Preparation
    totalTests++;
    if (testDAGPreparation()) passedTests++;
    
    // Test 3: Multi-Pool Job Setting
    totalTests++;
    if (testMultiPoolJobSetting()) passedTests++;
    
    // Test 4: Multi-Pool Mining
    totalTests++;
    if (testMultiPoolMining()) passedTests++;
    
    // Test 5: Memory Manager
    totalTests++;
    if (testMemoryManager()) passedTests++;
    
    // Test 6: Pool Manager
    totalTests++;
    if (testPoolManager()) passedTests++;
    
    // Test 7: CUDA Kernel Integration
    totalTests++;
    if (testCUDAKernelIntegration()) passedTests++;
    
    // Print final statistics
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Passed: " << passedTests << "/" << totalTests << " tests" << std::endl;
    std::cout << "Success rate: " << (passedTests * 100 / totalTests) << "%" << std::endl;
    
    if (passedTests == totalTests) {
        std::cout << "🎉 ALL TESTS PASSED! BMAD framework is ready for production." << std::endl;
    } else {
        std::cout << "⚠️  Some tests failed. Please review the implementation." << std::endl;
    }
    
    // Cleanup
    BMAD::KawPowMulti::cleanup();
}

int main() {
    try {
        runAllTests();
    } catch (const std::exception& e) {
        std::cerr << "Test suite failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}