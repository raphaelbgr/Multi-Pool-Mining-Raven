#ifndef BMAD_GPU_MEMORY_MANAGER_H
#define BMAD_GPU_MEMORY_MANAGER_H

#include "bmad_types.h"
#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace BMAD {

// GPU memory allocation status
enum class GPUAllocationStatus {
    SUCCESS,
    OUT_OF_MEMORY,
    INVALID_DEVICE,
    TRANSFER_FAILED,
    CLEANUP_FAILED
};

// GPU memory structure
struct GPUMemory {
    void* dag_memory;
    void* job_blobs_memory;
    void* results_memory;
    void* cache_memory;
    size_t dag_size;
    size_t job_blobs_size;
    size_t results_size;
    size_t cache_size;
    bool allocated;
    
    GPUMemory() : dag_memory(nullptr), job_blobs_memory(nullptr), 
                   results_memory(nullptr), cache_memory(nullptr),
                   dag_size(0), job_blobs_size(0), results_size(0), cache_size(0),
                   allocated(false) {}
};

// GPU memory manager for CUDA operations
class GPUMemoryManager {
public:
    GPUMemoryManager();
    ~GPUMemoryManager();
    
    // Initialize GPU memory manager
    GPUAllocationStatus initialize(uint32_t device_id);
    
    // GPU memory allocation
    GPUAllocationStatus allocateDAGMemory(size_t dag_size);
    GPUAllocationStatus allocateJobBlobsMemory(size_t job_count, size_t blob_size);
    GPUAllocationStatus allocateResultsMemory(size_t max_results);
    GPUAllocationStatus allocateCacheMemory(size_t cache_size);
    
    // Memory transfer to GPU
    GPUAllocationStatus copyDAGToGPU(const uint8_t* host_dag, size_t dag_size);
    GPUAllocationStatus copyJobBlobsToGPU(const std::vector<MultiPoolJob>& jobs);
    GPUAllocationStatus copyCacheToGPU(const uint8_t* host_cache, size_t cache_size);
    
    // Memory transfer from GPU
    GPUAllocationStatus copyResultsFromGPU(uint32_t* host_results, uint32_t* host_nonces, 
                                          uint32_t* result_count, size_t max_results);
    
    // Memory cleanup
    GPUAllocationStatus cleanup();
    
    // Get GPU memory pointers
    void* getDAGMemory() const { return m_gpu_memory.dag_memory; }
    void* getJobBlobsMemory() const { return m_gpu_memory.job_blobs_memory; }
    void* getResultsMemory() const { return m_gpu_memory.results_memory; }
    void* getCacheMemory() const { return m_gpu_memory.cache_memory; }
    
    // Get memory sizes
    size_t getDAGSize() const { return m_gpu_memory.dag_size; }
    size_t getJobBlobsSize() const { return m_gpu_memory.job_blobs_size; }
    size_t getResultsSize() const { return m_gpu_memory.results_size; }
    size_t getCacheSize() const { return m_gpu_memory.cache_size; }
    
    // Check if memory is allocated
    bool isAllocated() const { return m_gpu_memory.allocated; }
    
    // Error handling
    std::string getLastError() const { return m_last_error; }
    void clearError() { m_last_error.clear(); }
    
    // Memory statistics
    size_t getTotalAllocatedMemory() const;
    void printMemoryStats() const;

private:
    GPUMemory m_gpu_memory;
    uint32_t m_device_id;
    std::string m_last_error;
    bool m_initialized;
    
    // Helper functions
    GPUAllocationStatus allocateGPUMemory(void** gpu_ptr, size_t size, const std::string& description);
    GPUAllocationStatus copyToGPU(void* gpu_ptr, const void* host_ptr, size_t size, const std::string& description);
    GPUAllocationStatus copyFromGPU(void* host_ptr, const void* gpu_ptr, size_t size, const std::string& description);
    GPUAllocationStatus freeGPUMemory(void* gpu_ptr, const std::string& description);
    
    // Error handling
    void setError(const std::string& error);
    void logAllocation(const std::string& description, size_t size);
    void logTransfer(const std::string& description, size_t size);
};

} // namespace BMAD

#endif // BMAD_GPU_MEMORY_MANAGER_H