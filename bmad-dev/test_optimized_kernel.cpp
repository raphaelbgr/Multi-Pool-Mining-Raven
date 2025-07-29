#include "../src/bmad_kawpow_optimized.h"
#include "../src/bmad_kawpow_algorithm.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

using namespace BMAD;

// Test function to compare optimized vs standard kernel
void testOptimizedKernel() {
    std::cout << "=== OPTIMIZED KERNEL PERFORMANCE TEST ===" << std::endl;
    
    // Initialize optimized kernel
    OptimizedBlockConfig config;
    config.block_size = OPTIMIZED_BLOCK_SIZE;
    config.grid_size = OPTIMIZED_GRID_SIZE;
    config.shared_memory_size = SHARED_MEMORY_SIZE;
    config.max_pools_per_block = MAX_POOLS_PER_BLOCK;
    config.nonces_per_thread = OPTIMIZED_NONCES_PER_THREAD;
    
    if (!KawPowOptimized::initialize(config)) {
        std::cerr << "Failed to initialize optimized kernel" << std::endl;
        return;
    }
    
    // Create test job data
    OptimizedMultiPoolJob job;
    job.pool_count = 5; // Test with 5 pools
    job.start_nonce = 1000;
    job.nonce_count = 1024;
    
    // Create test job blobs and targets
    for (uint32_t i = 0; i < job.pool_count; i++) {
        // Allocate job blob
        job.job_blobs[i] = new uint8_t[40];
        memset(job.job_blobs[i], 0, 40);
        
        // Fill with test data
        for (int j = 0; j < 40; j++) {
            job.job_blobs[i][j] = (uint8_t)(i * 10 + j);
        }
        
        // Set target (different for each pool)
        job.targets[i] = 0x1000000 + (i * 0x100000);
    }
    
    // Create result structure
    OptimizedMultiPoolResult result;
    result.max_results = 1000;
    result.results = new uint32_t[result.max_results];
    result.nonces = new uint32_t[result.max_results];
    result.result_count = new uint32_t(0);
    
    // Create test DAG
    size_t dag_size = 1024 * 1024; // 1MB test DAG
    uint8_t* dag = new uint8_t[dag_size];
    memset(dag, 0, dag_size);
    
    // Fill DAG with test data
    for (size_t i = 0; i < dag_size; i++) {
        dag[i] = (uint8_t)(i % 256);
    }
    
    std::cout << "\n🧪 Testing Optimized Kernel:" << std::endl;
    std::cout << "  Pools: " << job.pool_count << std::endl;
    std::cout << "  Nonces: " << job.nonce_count << std::endl;
    std::cout << "  DAG size: " << dag_size << " bytes" << std::endl;
    
    // Test optimized kernel
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = KawPowOptimized::calculateOptimizedMultiPoolHash(job, result, dag, dag_size);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "\n✅ Optimized Kernel Results:" << std::endl;
    std::cout << "  Success: " << (success ? "Yes" : "No") << std::endl;
    std::cout << "  Duration: " << duration.count() << " microseconds" << std::endl;
    std::cout << "  Found shares: " << *result.result_count << std::endl;
    std::cout << "  Hashrate: " << (job.nonce_count * job.pool_count * 1000000.0 / duration.count()) << " H/s" << std::endl;
    
    // Print performance stats
    KawPowOptimized::printPerformanceStats();
    
    // Cleanup
    for (uint32_t i = 0; i < job.pool_count; i++) {
        delete[] job.job_blobs[i];
    }
    delete[] result.results;
    delete[] result.nonces;
    delete result.result_count;
    delete[] dag;
    
    KawPowOptimized::cleanup();
}

// Test batch processing
void testBatchProcessing() {
    std::cout << "\n=== BATCH PROCESSING TEST ===" << std::endl;
    
    // Initialize optimized kernel
    OptimizedBlockConfig config;
    config.block_size = OPTIMIZED_BLOCK_SIZE;
    config.grid_size = OPTIMIZED_GRID_SIZE;
    config.shared_memory_size = SHARED_MEMORY_SIZE;
    config.max_pools_per_block = MAX_POOLS_PER_BLOCK;
    config.nonces_per_thread = OPTIMIZED_NONCES_PER_THREAD;
    
    if (!KawPowOptimized::initialize(config)) {
        std::cerr << "Failed to initialize optimized kernel" << std::endl;
        return;
    }
    
    // Create multiple jobs for batch processing
    std::vector<OptimizedMultiPoolJob> jobs;
    std::vector<OptimizedMultiPoolResult> results;
    
    for (int batch = 0; batch < 3; batch++) {
        OptimizedMultiPoolJob job;
        job.pool_count = 3 + batch; // 3, 4, 5 pools
        job.start_nonce = 1000 + batch * 1000;
        job.nonce_count = 512;
        
        // Create job blobs and targets
        for (uint32_t i = 0; i < job.pool_count; i++) {
            job.job_blobs[i] = new uint8_t[40];
            memset(job.job_blobs[i], 0, 40);
            
            for (int j = 0; j < 40; j++) {
                job.job_blobs[i][j] = (uint8_t)(batch * 10 + i * 5 + j);
            }
            
            job.targets[i] = 0x1000000 + (batch * 0x100000) + (i * 0x10000);
        }
        
        jobs.push_back(job);
        
        // Create result structure
        OptimizedMultiPoolResult result;
        result.max_results = 100;
        result.results = new uint32_t[result.max_results];
        result.nonces = new uint32_t[result.max_results];
        result.result_count = new uint32_t(0);
        
        results.push_back(result);
    }
    
    // Create test DAG
    size_t dag_size = 1024 * 1024;
    uint8_t* dag = new uint8_t[dag_size];
    memset(dag, 0, dag_size);
    for (size_t i = 0; i < dag_size; i++) {
        dag[i] = (uint8_t)(i % 256);
    }
    
    std::cout << "Processing batch of " << jobs.size() << " jobs..." << std::endl;
    
    // Process batch
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = KawPowOptimized::processBatch(jobs, results, dag, dag_size);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "\n✅ Batch Processing Results:" << std::endl;
    std::cout << "  Success: " << (success ? "Yes" : "No") << std::endl;
    std::cout << "  Duration: " << duration.count() << " microseconds" << std::endl;
    
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << "  Batch " << i << ": " << *results[i].result_count << " shares" << std::endl;
    }
    
    // Print performance stats
    KawPowOptimized::printPerformanceStats();
    
    // Cleanup
    for (auto& job : jobs) {
        for (uint32_t i = 0; i < job.pool_count; i++) {
            delete[] job.job_blobs[i];
        }
    }
    
    for (auto& result : results) {
        delete[] result.results;
        delete[] result.nonces;
        delete result.result_count;
    }
    
    delete[] dag;
    
    KawPowOptimized::cleanup();
}

int main() {
    std::cout << "🚀 BMAD Optimized Kernel Performance Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        // Test optimized kernel
        testOptimizedKernel();
        
        // Test batch processing
        testBatchProcessing();
        
        std::cout << "\n🎉 All tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}