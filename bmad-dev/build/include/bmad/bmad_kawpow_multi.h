#ifndef BMAD_KAWPOW_MULTI_H
#define BMAD_KAWPOW_MULTI_H

#include "bmad_types.h"
#include "bmad_memory_manager.h"
#include <vector>
#include <memory>

namespace BMAD {

class KawPowMulti {
public:
    KawPowMulti();
    ~KawPowMulti();

    // Initialize multi-pool KawPow miner
    bool initialize(const BMADConfig& config, uint32_t device_id);
    
    // Prepare DAG and cache for mining
    bool prepare(const void* cache, size_t cache_size, size_t dag_size, uint32_t height);
    
    // Set multi-pool jobs
    bool setJobs(const std::vector<MultiPoolJob>& jobs);
    
    // Mine with multiple pools simultaneously
    bool mine(uint32_t start_nonce, uint32_t num_nonces, 
              std::vector<MultiPoolResult>& results);
    
    // Get mining statistics
    uint32_t getProcessedHashes() const { return m_processed_hashes; }
    uint32_t getSkippedHashes() const { return m_skipped_hashes; }
    
    // Cleanup
    void cleanup();
    
    // Get device info
    uint32_t getDeviceId() const { return m_device_id; }
    uint32_t getBlocks() const { return m_blocks; }
    uint32_t getThreads() const { return m_threads; }
    uint32_t getIntensity() const { return m_intensity; }

private:
    // Memory manager
    std::unique_ptr<MemoryManager> m_memory_manager;
    
    // Mining context
    BMADContext m_context;
    BMADConfig m_config;
    
    // Device parameters
    uint32_t m_device_id;
    uint32_t m_blocks;
    uint32_t m_threads;
    uint32_t m_intensity;
    
    // Statistics
    uint32_t m_processed_hashes;
    uint32_t m_skipped_hashes;
    
    // CUDA kernel handles
    void* m_kernel_handle;
    void* m_module_handle;
    
    // Internal methods
    bool loadKernel();
    bool setupKernelParameters();
    bool launchKernel(uint32_t start_nonce, uint32_t num_nonces);
    bool processResults(std::vector<MultiPoolResult>& results);
    
    // Utility methods
    bool validateJob(const MultiPoolJob& job);
    bool updateJob(const MultiPoolJob& job);
    void resetStatistics();
};

// CUDA kernel declarations (implemented in .cu file)
extern "C" {
    // Multi-pool KawPow kernel
    __global__ void kawpow_multi_search(
        const void* dag,
        const uint8_t* job_blobs,
        const uint64_t* targets,
        const uint32_t num_pools,
        const uint32_t start_nonce,
        const uint32_t num_nonces,
        uint32_t* results,
        uint32_t* result_count,
        uint32_t* skipped_hashes
    );
    
    // DAG calculation kernel
    __global__ void kawpow_calculate_dag(
        uint64_t* dag,
        const uint64_t* cache,
        const uint32_t dag_size,
        const uint32_t cache_size
    );
}

} // namespace BMAD

#endif // BMAD_KAWPOW_MULTI_H