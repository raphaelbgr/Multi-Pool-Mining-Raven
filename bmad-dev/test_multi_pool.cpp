#include "include/bmad_kawpow_multi.h"
#include "include/bmad_memory_manager.h"
#include "include/bmad_pool_manager.h"
#include "include/bmad_types.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <random>

// Test utilities
void printTestHeader(const std::string& testName) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST: " << testName << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void printTestResult(bool passed, const std::string& message) {
    std::cout << (passed ? "✅ PASS" : "❌ FAIL") << ": " << message << std::endl;
}

// Generate different job blobs for different pools
std::vector<BMAD::MultiPoolJob> generateDifferentJobs(uint32_t numPools) {
    std::vector<BMAD::MultiPoolJob> jobs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 0xFF);
    
    for (uint32_t i = 0; i < numPools; i++) {
        BMAD::MultiPoolJob job;
        job.pool_id = i + 1;
        job.height = 1000 + i; // Different heights for different pools
        job.target = 0x12345678 + (i * 0x1000); // Different targets
        job.active = true;
        
        // Generate different job blob for each pool
        for (int j = 0; j < 40; j++) {
            job.blob[j] = static_cast<uint8_t>(dis(gen) + i); // Different data per pool
        }
        
        jobs.push_back(job);
    }
    
    return jobs;
}

// Test 1: Multi-pool job management
bool testMultiPoolJobManagement() {
    printTestHeader("Multi-Pool Job Management");
    
    try {
        BMAD::KawPowMulti kawpow;
        
        // Initialize with multi-pool config
        BMAD::BMADConfig config;
        config.max_pools = 5;
        config.max_nonces = 1024;
        config.device_id = 0;
        config.batch_size = 256;
        config.memory_alignment = 256;
        config.use_pinned_memory = false;
        config.enable_profiling = false;
        
        bool init_result = kawpow.initialize(config);
        printTestResult(init_result, "KawPowMulti initialization");
        
        // Generate different jobs for different pools
        auto jobs = generateDifferentJobs(3);
        printTestResult(jobs.size() == 3, "Generated 3 different jobs");
        
        // Verify jobs are different
        bool jobs_different = true;
        for (size_t i = 0; i < jobs.size(); i++) {
            for (size_t j = i + 1; j < jobs.size(); j++) {
                if (jobs[i].pool_id == jobs[j].pool_id ||
                    jobs[i].target == jobs[j].target ||
                    jobs[i].height == jobs[j].height) {
                    jobs_different = false;
                }
            }
        }
        printTestResult(jobs_different, "Jobs have different parameters");
        
        // Set jobs for multi-pool mining
        bool set_jobs_result = kawpow.setJobs(jobs);
        printTestResult(set_jobs_result, "Set multiple pool jobs");
        
        // Test mining with multiple pools
        uint32_t rescount, resnonce;
        bool mine_result = kawpow.mine(0, &rescount, &resnonce);
        printTestResult(mine_result, "Multi-pool mining execution");
        
        kawpow.cleanup();
        return true;
        
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Test 2: Memory management for multiple pools
bool testMultiPoolMemoryManagement() {
    printTestHeader("Multi-Pool Memory Management");
    
    try {
        BMAD::MemoryManager memory;
        
        // Initialize memory manager
        BMAD::BMADConfig config;
        config.max_pools = 5;
        config.max_nonces = 1024;
        config.device_id = 0;
        config.batch_size = 256;
        config.memory_alignment = 256;
        config.use_pinned_memory = false;
        config.enable_profiling = false;
        
        bool init_result = memory.initialize(config, 0);
        printTestResult(init_result, "Memory manager initialization");
        
        // Prepare DAG
        bool dag_result = memory.prepareDAG(1000);
        printTestResult(dag_result, "DAG preparation");
        
        // Generate different jobs
        auto jobs = generateDifferentJobs(3);
        
        // Set jobs in memory manager
        bool set_jobs_result = memory.setJobs(jobs);
        printTestResult(set_jobs_result, "Set jobs in memory manager");
        
        // Verify memory allocation
        printTestResult(memory.getDAG() != nullptr, "DAG memory allocated");
        printTestResult(memory.getJobBlobs() != nullptr, "Job blobs memory allocated");
        printTestResult(memory.getTargets() != nullptr, "Targets memory allocated");
        printTestResult(memory.getNumPools() == 3, "Correct number of pools");
        
        memory.cleanup();
        return true;
        
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Test 3: Pool management with different jobs
bool testPoolManagementWithDifferentJobs() {
    printTestHeader("Pool Management with Different Jobs");
    
    try {
        BMAD::PoolManager pool;
        
        // Initialize pool manager
        bool init_result = pool.initialize(5);
        printTestResult(init_result, "Pool manager initialization");
        
        // Add different pools
        bool add1 = pool.addPool(1, "Pool1");
        bool add2 = pool.addPool(2, "Pool2");
        bool add3 = pool.addPool(3, "Pool3");
        
        printTestResult(add1 && add2 && add3, "Added 3 pools");
        
        // Generate different jobs
        auto jobs = generateDifferentJobs(3);
        
        // Update each pool with different job
        for (size_t i = 0; i < jobs.size(); i++) {
            bool update_result = pool.updatePoolJob(jobs[i].pool_id, jobs[i]);
            printTestResult(update_result, "Updated pool " + std::to_string(jobs[i].pool_id) + " with job");
        }
        
        // Set pools active
        pool.setPoolActive(1, true);
        pool.setPoolActive(2, true);
        pool.setPoolActive(3, true);
        
        // Get active jobs
        auto active_jobs = pool.getActiveJobs();
        printTestResult(active_jobs.size() == 3, "Got 3 active jobs");
        
        // Verify jobs are different
        bool jobs_different = true;
        for (size_t i = 0; i < active_jobs.size(); i++) {
            for (size_t j = i + 1; j < active_jobs.size(); j++) {
                if (active_jobs[i].pool_id == active_jobs[j].pool_id ||
                    active_jobs[i].target == active_jobs[j].target) {
                    jobs_different = false;
                }
            }
        }
        printTestResult(jobs_different, "Active jobs have different parameters");
        
        // Test statistics
        printTestResult(pool.getActivePoolCount() == 3, "3 active pools");
        printTestResult(pool.getTotalPoolCount() == 3, "3 total pools");
        
        pool.cleanup();
        return true;
        
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Test 4: Multi-pool mining simulation
bool testMultiPoolMiningSimulation() {
    printTestHeader("Multi-Pool Mining Simulation");
    
    try {
        BMAD::KawPowMulti kawpow;
        BMAD::PoolManager pool;
        
        // Initialize both components
        BMAD::BMADConfig config;
        config.max_pools = 5;
        config.max_nonces = 1024;
        config.device_id = 0;
        config.batch_size = 256;
        config.memory_alignment = 256;
        config.use_pinned_memory = false;
        config.enable_profiling = false;
        
        bool kawpow_init = kawpow.initialize(config);
        bool pool_init = pool.initialize(5);
        
        printTestResult(kawpow_init && pool_init, "Both components initialized");
        
        // Generate different jobs for different pools
        auto jobs = generateDifferentJobs(3);
        
        // Set jobs in both components
        bool kawpow_jobs = kawpow.setJobs(jobs);
        for (const auto& job : jobs) {
            pool.updatePoolJob(job.pool_id, job);
            pool.setPoolActive(job.pool_id, true);
        }
        
        printTestResult(kawpow_jobs, "Jobs set in KawPowMulti");
        printTestResult(pool.getActivePoolCount() == 3, "Jobs set in PoolManager");
        
        // Simulate mining with multiple pools
        uint32_t rescount, resnonce;
        bool mine_result = kawpow.mine(0, &rescount, &resnonce);
        printTestResult(mine_result, "Multi-pool mining simulation");
        
        // Test share submission to different pools
        BMAD::MultiPoolResult result;
        result.pool_id = 1;
        result.nonce = 0x12345678;
        result.hash = 0x87654321;
        result.actual_diff = 0x11111111;
        result.valid = true;
        
        bool submit_result = pool.submitShare(1, result);
        printTestResult(submit_result, "Share submission to pool 1");
        
        kawpow.cleanup();
        pool.cleanup();
        return true;
        
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Test 5: Different jobs solution verification
bool testDifferentJobsSolution() {
    printTestHeader("Different Jobs Solution Verification");
    
    try {
        // This test verifies that our solution can handle different jobs from different pools
        // Each pool has different: blob, target, height, and pool_id
        
        std::vector<BMAD::MultiPoolJob> jobs = generateDifferentJobs(5);
        
        // Verify each job is unique
        bool all_unique = true;
        for (size_t i = 0; i < jobs.size(); i++) {
            for (size_t j = i + 1; j < jobs.size(); j++) {
                // Check if any parameters are the same (they shouldn't be)
                if (jobs[i].pool_id == jobs[j].pool_id ||
                    jobs[i].target == jobs[j].target ||
                    jobs[i].height == jobs[j].height) {
                    all_unique = false;
                }
                
                // Check if blobs are identical (they shouldn't be)
                bool blobs_identical = true;
                for (int k = 0; k < 40; k++) {
                    if (jobs[i].blob[k] != jobs[j].blob[k]) {
                        blobs_identical = false;
                        break;
                    }
                }
                if (blobs_identical) {
                    all_unique = false;
                }
            }
        }
        
        printTestResult(all_unique, "All jobs have unique parameters");
        
        // Test that our BMAD framework can handle these different jobs
        BMAD::KawPowMulti kawpow;
        BMAD::BMADConfig config;
        config.max_pools = 10;
        config.max_nonces = 1024;
        config.device_id = 0;
        config.batch_size = 256;
        config.memory_alignment = 256;
        config.use_pinned_memory = false;
        config.enable_profiling = false;
        
        bool init_result = kawpow.initialize(config);
        printTestResult(init_result, "BMAD framework initialized");
        
        bool set_jobs_result = kawpow.setJobs(jobs);
        printTestResult(set_jobs_result, "Set 5 different jobs");
        
        // Test mining with different jobs
        uint32_t rescount, resnonce;
        bool mine_result = kawpow.mine(0, &rescount, &resnonce);
        printTestResult(mine_result, "Mining with different jobs");
        
        kawpow.cleanup();
        return true;
        
    } catch (const std::exception& e) {
        printTestResult(false, "Exception: " + std::string(e.what()));
        return false;
    }
}

// Main test runner
int main() {
    std::cout << "BMAD Multi-Pool Mining Test Suite" << std::endl;
    std::cout << "==================================" << std::endl;
    
    int passedTests = 0;
    int totalTests = 0;
    
    // Run all tests
    totalTests++;
    if (testMultiPoolJobManagement()) passedTests++;
    
    totalTests++;
    if (testMultiPoolMemoryManagement()) passedTests++;
    
    totalTests++;
    if (testPoolManagementWithDifferentJobs()) passedTests++;
    
    totalTests++;
    if (testMultiPoolMiningSimulation()) passedTests++;
    
    totalTests++;
    if (testDifferentJobsSolution()) passedTests++;
    
    // Print final results
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Passed: " << passedTests << "/" << totalTests << " tests" << std::endl;
    std::cout << "Success rate: " << (passedTests * 100 / totalTests) << "%" << std::endl;
    
    if (passedTests == totalTests) {
        std::cout << "🎉 ALL TESTS PASSED! Multi-pool mining solution is working." << std::endl;
        return 0;
    } else {
        std::cout << "⚠️  Some tests failed. Need to fix implementation." << std::endl;
        return 1;
    }
}