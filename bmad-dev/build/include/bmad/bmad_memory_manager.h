#ifndef BMAD_MEMORY_MANAGER_H
#define BMAD_MEMORY_MANAGER_H

#include "bmad_types.h"
#include <cuda_runtime.h>
#include <memory>

namespace BMAD {

class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();

    // Initialize memory manager
    bool initialize(const BMADConfig& config, uint32_t device_id);
    
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
    
    // Get device memory pointers
    void* getDAG() const { return m_memory.dag; }
    void* getCache() const { return m_memory.cache; }
    void* getJobBlobs() const { return m_memory.job_blobs; }
    void* getTargets() const { return m_memory.targets; }
    void* getResults() const { return m_memory.results; }
    
    // Get memory info
    size_t getDAGSize() const { return m_memory.dag_size; }
    size_t getCacheSize() const { return m_memory.cache_size; }
    uint32_t getNumPools() const { return m_memory.num_pools; }
    
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