#ifndef BMAD_MEMORY_MANAGER_H
#define BMAD_MEMORY_MANAGER_H

#include "bmad_types.h"
// #include <cuda_runtime.h>  // Commented out for initial compilation
#include <memory>

namespace BMAD {

class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();

    // Initialize memory manager
    bool initialize(const BMADConfig& config, uint32_t device_id);
    
    // Prepare DAG for mining
    bool prepareDAG(const uint64_t height);
    
    // Set jobs for multi-pool mining
    bool setJobs(const std::vector<MultiPoolJob>& jobs);
    
    // Allocate DAG memory
    bool allocateDAG(size_t dag_size);
    
    // Allocate cache memory
    bool allocateCache(size_t cache_size);
    
    // Allocate multi-pool job memory
    bool allocateJobMemory(uint32_t num_pools);
    
    // Allocate results memory
    bool allocateResultsMemory(uint32_t max_results);
    
    // Copy DAG data to device
    bool copyDAGToDevice(const void* host_dag, size_t dag_size);
    
    // Copy cache data to device
    bool copyCacheToDevice(const void* host_cache, size_t cache_size);
    
    // Copy job data to device
    bool copyJobsToDevice(const std::vector<MultiPoolJob>& jobs);
    
    // Get memory access methods
    void* getDAG() const { return m_memory.dag; }
    size_t getDAGSize() const { return m_memory.dag_size; }
    void* getCache() const { return m_memory.cache; }
    size_t getCacheSize() const { return m_memory.cache_size; }
    void* getJobBlobs() const { return m_memory.job_blobs; }
    void* getTargets() const { return m_memory.targets; }
    void* getResults() const { return m_memory.results; }
    uint32_t getNumPools() const { return m_memory.num_pools; }
    uint32_t getMaxPools() const { return m_memory.max_pools; }
    
    // Cleanup
    void cleanup();
    
    // Memory alignment utilities
    static size_t alignSize(size_t size, size_t alignment);
    static void* alignPointer(void* ptr, size_t alignment);

private:
    BMADMemory m_memory;
    BMADConfig m_config;
    uint32_t m_device_id;
    bool m_initialized;
    
    // Pinned memory for faster transfers
    void* m_pinned_job_blobs;
    void* m_pinned_targets;
    void* m_pinned_results;
    
    // Memory allocation helpers
    bool allocateDeviceMemory(void** ptr, size_t size);
    bool allocatePinnedMemory(void** ptr, size_t size);
    void freeDeviceMemory(void* ptr);
    void freePinnedMemory(void* ptr);
};

} // namespace BMAD

#endif // BMAD_MEMORY_MANAGER_H