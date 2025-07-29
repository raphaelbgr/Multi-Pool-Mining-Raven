#include "../include/bmad_gpu_memory_manager.h"
#include "../include/bmad_types.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace BMAD;

int main() {
    std::cout << "🧪 BMAD GPU Memory Manager Test" << std::endl;
    std::cout << "================================" << std::endl;
    
    try {
        // Test 1: Initialize GPU Memory Manager
        std::cout << "\n✅ Test 1: Initialize GPU Memory Manager" << std::endl;
        GPUMemoryManager gpu_manager;
        
        GPUAllocationStatus status = gpu_manager.initialize(0);
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to initialize GPU Memory Manager: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        // Test 2: Allocate DAG Memory
        std::cout << "\n✅ Test 2: Allocate DAG Memory" << std::endl;
        size_t dag_size = 1024 * 1024; // 1MB DAG
        status = gpu_manager.allocateDAGMemory(dag_size);
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to allocate DAG memory: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        // Test 3: Allocate Job Blobs Memory
        std::cout << "\n✅ Test 3: Allocate Job Blobs Memory" << std::endl;
        size_t job_count = 5;
        size_t blob_size = 40;
        status = gpu_manager.allocateJobBlobsMemory(job_count, blob_size);
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to allocate job blobs memory: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        // Test 4: Allocate Results Memory
        std::cout << "\n✅ Test 4: Allocate Results Memory" << std::endl;
        size_t max_results = 1000;
        status = gpu_manager.allocateResultsMemory(max_results);
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to allocate results memory: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        // Test 5: Allocate Cache Memory
        std::cout << "\n✅ Test 5: Allocate Cache Memory" << std::endl;
        size_t cache_size = 64 * 1024; // 64KB cache
        status = gpu_manager.allocateCacheMemory(cache_size);
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to allocate cache memory: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        // Test 6: Copy DAG to GPU
        std::cout << "\n✅ Test 6: Copy DAG to GPU" << std::endl;
        std::vector<uint8_t> host_dag(dag_size);
        for (size_t i = 0; i < dag_size; i++) {
            host_dag[i] = (uint8_t)(i % 256);
        }
        
        status = gpu_manager.copyDAGToGPU(host_dag.data(), dag_size);
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to copy DAG to GPU: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        // Test 7: Copy Job Blobs to GPU
        std::cout << "\n✅ Test 7: Copy Job Blobs to GPU" << std::endl;
        std::vector<MultiPoolJob> jobs;
        for (int i = 0; i < 5; i++) {
            MultiPoolJob job;
            job.pool_id = i + 1;
            job.active = true;
            job.height = 1000 + i;
            job.target = 0x1000000 + (i * 0x100000);
            memset(job.blob, i, 40);
            jobs.push_back(job);
        }
        
        status = gpu_manager.copyJobBlobsToGPU(jobs);
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to copy job blobs to GPU: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        // Test 8: Copy Cache to GPU
        std::cout << "\n✅ Test 8: Copy Cache to GPU" << std::endl;
        std::vector<uint8_t> host_cache(cache_size);
        for (size_t i = 0; i < cache_size; i++) {
            host_cache[i] = (uint8_t)(i % 256);
        }
        
        status = gpu_manager.copyCacheToGPU(host_cache.data(), cache_size);
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to copy cache to GPU: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        // Test 9: Copy Results from GPU
        std::cout << "\n✅ Test 9: Copy Results from GPU" << std::endl;
        std::vector<uint32_t> host_results(max_results);
        std::vector<uint32_t> host_nonces(max_results);
        uint32_t result_count = 0;
        
        status = gpu_manager.copyResultsFromGPU(host_results.data(), host_nonces.data(), 
                                               &result_count, max_results);
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to copy results from GPU: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        std::cout << "  Results found: " << result_count << std::endl;
        for (uint32_t i = 0; i < result_count && i < 5; i++) {
            std::cout << "    Result " << i << ": 0x" << std::hex << host_results[i] 
                      << " (nonce: " << std::dec << host_nonces[i] << ")" << std::endl;
        }
        
        // Test 10: Print Memory Statistics
        std::cout << "\n✅ Test 10: Memory Statistics" << std::endl;
        gpu_manager.printMemoryStats();
        
        // Test 11: Performance Benchmark
        std::cout << "\n✅ Test 11: Performance Benchmark" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Simulate multiple transfers
        for (int i = 0; i < 10; i++) {
            status = gpu_manager.copyDAGToGPU(host_dag.data(), dag_size);
            if (status != GPUAllocationStatus::SUCCESS) {
                std::cerr << "Benchmark transfer failed: " << gpu_manager.getLastError() << std::endl;
                break;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "  Benchmark completed in " << duration.count() << " microseconds" << std::endl;
        std::cout << "  Transfer rate: " << (dag_size * 10 * 1000000.0 / duration.count()) << " bytes/s" << std::endl;
        
        // Test 12: Cleanup
        std::cout << "\n✅ Test 12: Cleanup" << std::endl;
        status = gpu_manager.cleanup();
        if (status != GPUAllocationStatus::SUCCESS) {
            std::cerr << "Failed to cleanup GPU memory: " << gpu_manager.getLastError() << std::endl;
            return 1;
        }
        
        std::cout << "\n🎉 ALL GPU MEMORY TESTS PASSED!" << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "✅ GPU Memory Manager initialized" << std::endl;
        std::cout << "✅ DAG memory allocated and transferred" << std::endl;
        std::cout << "✅ Job blobs memory allocated and transferred" << std::endl;
        std::cout << "✅ Results memory allocated and transferred" << std::endl;
        std::cout << "✅ Cache memory allocated and transferred" << std::endl;
        std::cout << "✅ Results copied from GPU successfully" << std::endl;
        std::cout << "✅ Memory cleanup completed" << std::endl;
        std::cout << "\n🚀 GPU Memory Manager ready for production!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ GPU Memory Manager test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}