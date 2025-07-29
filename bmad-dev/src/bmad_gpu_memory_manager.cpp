#include "bmad_gpu_memory_manager.h"
#include <iostream>
#include <cstring>
#include <sstream>

namespace BMAD {

// Constructor
GPUMemoryManager::GPUMemoryManager() 
    : m_device_id(0), m_initialized(false) {
    std::cout << "GPU Memory Manager created" << std::endl;
}

// Destructor
GPUMemoryManager::~GPUMemoryManager() {
    cleanup();
    std::cout << "GPU Memory Manager destroyed" << std::endl;
}

// Initialize GPU memory manager
GPUAllocationStatus GPUMemoryManager::initialize(uint32_t device_id) {
    m_device_id = device_id;
    m_initialized = true;
    
    std::cout << "GPU Memory Manager initialized for device " << device_id << std::endl;
    
    // For now, simulate GPU initialization
    // In real implementation, this would call cudaSetDevice(device_id)
    
    return GPUAllocationStatus::SUCCESS;
}

// Allocate GPU memory
GPUAllocationStatus GPUMemoryManager::allocateGPUMemory(void** gpu_ptr, size_t size, const std::string& description) {
    if (!m_initialized) {
        setError("GPU Memory Manager not initialized");
        return GPUAllocationStatus::INVALID_DEVICE;
    }
    
    // For now, simulate GPU memory allocation with host memory
    // In real implementation, this would call cudaMalloc(gpu_ptr, size)
    *gpu_ptr = malloc(size);
    
    if (*gpu_ptr == nullptr) {
        setError("Failed to allocate " + std::to_string(size) + " bytes for " + description);
        return GPUAllocationStatus::OUT_OF_MEMORY;
    }
    
    logAllocation(description, size);
    return GPUAllocationStatus::SUCCESS;
}

// Allocate DAG memory
GPUAllocationStatus GPUMemoryManager::allocateDAGMemory(size_t dag_size) {
    GPUAllocationStatus status = allocateGPUMemory(&m_gpu_memory.dag_memory, dag_size, "DAG");
    if (status == GPUAllocationStatus::SUCCESS) {
        m_gpu_memory.dag_size = dag_size;
        m_gpu_memory.allocated = true;
    }
    return status;
}

// Allocate job blobs memory
GPUAllocationStatus GPUMemoryManager::allocateJobBlobsMemory(size_t job_count, size_t blob_size) {
    size_t total_size = job_count * blob_size;
    GPUAllocationStatus status = allocateGPUMemory(&m_gpu_memory.job_blobs_memory, total_size, "Job Blobs");
    if (status == GPUAllocationStatus::SUCCESS) {
        m_gpu_memory.job_blobs_size = total_size;
    }
    return status;
}

// Allocate results memory
GPUAllocationStatus GPUMemoryManager::allocateResultsMemory(size_t max_results) {
    size_t total_size = max_results * sizeof(uint32_t) * 2; // results + nonces
    GPUAllocationStatus status = allocateGPUMemory(&m_gpu_memory.results_memory, total_size, "Results");
    if (status == GPUAllocationStatus::SUCCESS) {
        m_gpu_memory.results_size = total_size;
    }
    return status;
}

// Allocate cache memory
GPUAllocationStatus GPUMemoryManager::allocateCacheMemory(size_t cache_size) {
    GPUAllocationStatus status = allocateGPUMemory(&m_gpu_memory.cache_memory, cache_size, "Cache");
    if (status == GPUAllocationStatus::SUCCESS) {
        m_gpu_memory.cache_size = cache_size;
    }
    return status;
}

// Copy memory to GPU
GPUAllocationStatus GPUMemoryManager::copyToGPU(void* gpu_ptr, const void* host_ptr, size_t size, const std::string& description) {
    if (!m_initialized) {
        setError("GPU Memory Manager not initialized");
        return GPUAllocationStatus::INVALID_DEVICE;
    }
    
    if (!gpu_ptr || !host_ptr) {
        setError("Invalid pointers for " + description + " transfer");
        return GPUAllocationStatus::TRANSFER_FAILED;
    }
    
    // For now, simulate GPU memory copy with host memory copy
    // In real implementation, this would call cudaMemcpy(gpu_ptr, host_ptr, size, cudaMemcpyHostToDevice)
    memcpy(gpu_ptr, host_ptr, size);
    
    logTransfer("Host to GPU: " + description, size);
    return GPUAllocationStatus::SUCCESS;
}

// Copy DAG to GPU
GPUAllocationStatus GPUMemoryManager::copyDAGToGPU(const uint8_t* host_dag, size_t dag_size) {
    if (!m_gpu_memory.dag_memory) {
        setError("DAG memory not allocated");
        return GPUAllocationStatus::TRANSFER_FAILED;
    }
    
    return copyToGPU(m_gpu_memory.dag_memory, host_dag, dag_size, "DAG");
}

// Copy job blobs to GPU
GPUAllocationStatus GPUMemoryManager::copyJobBlobsToGPU(const std::vector<MultiPoolJob>& jobs) {
    if (!m_gpu_memory.job_blobs_memory) {
        setError("Job blobs memory not allocated");
        return GPUAllocationStatus::TRANSFER_FAILED;
    }
    
    // Pack job blobs into contiguous memory
    uint8_t* job_blobs_data = static_cast<uint8_t*>(m_gpu_memory.job_blobs_memory);
    size_t offset = 0;
    
    for (const auto& job : jobs) {
        if (offset + sizeof(job.blob) <= m_gpu_memory.job_blobs_size) {
            memcpy(job_blobs_data + offset, job.blob, sizeof(job.blob));
            offset += sizeof(job.blob);
        }
    }
    
    logTransfer("Job Blobs to GPU", offset);
    return GPUAllocationStatus::SUCCESS;
}

// Copy cache to GPU
GPUAllocationStatus GPUMemoryManager::copyCacheToGPU(const uint8_t* host_cache, size_t cache_size) {
    if (!m_gpu_memory.cache_memory) {
        setError("Cache memory not allocated");
        return GPUAllocationStatus::TRANSFER_FAILED;
    }
    
    return copyToGPU(m_gpu_memory.cache_memory, host_cache, cache_size, "Cache");
}

// Copy memory from GPU
GPUAllocationStatus GPUMemoryManager::copyFromGPU(void* host_ptr, const void* gpu_ptr, size_t size, const std::string& description) {
    if (!m_initialized) {
        setError("GPU Memory Manager not initialized");
        return GPUAllocationStatus::INVALID_DEVICE;
    }
    
    if (!gpu_ptr || !host_ptr) {
        setError("Invalid pointers for " + description + " transfer");
        return GPUAllocationStatus::TRANSFER_FAILED;
    }
    
    // For now, simulate GPU memory copy with host memory copy
    // In real implementation, this would call cudaMemcpy(host_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost)
    memcpy(host_ptr, gpu_ptr, size);
    
    logTransfer("GPU to Host: " + description, size);
    return GPUAllocationStatus::SUCCESS;
}

// Copy results from GPU
GPUAllocationStatus GPUMemoryManager::copyResultsFromGPU(uint32_t* host_results, uint32_t* host_nonces, 
                                                        uint32_t* result_count, size_t max_results) {
    if (!m_gpu_memory.results_memory) {
        setError("Results memory not allocated");
        return GPUAllocationStatus::TRANSFER_FAILED;
    }
    
    // For now, simulate results
    // In real implementation, this would copy actual results from GPU
    *result_count = 3; // Simulate finding 3 shares
    for (size_t i = 0; i < *result_count && i < max_results; i++) {
        host_results[i] = 0x12345678 + i;
        host_nonces[i] = 1000 + i;
    }
    
    logTransfer("Results from GPU", *result_count * sizeof(uint32_t) * 2);
    return GPUAllocationStatus::SUCCESS;
}

// Free GPU memory
GPUAllocationStatus GPUMemoryManager::freeGPUMemory(void* gpu_ptr, const std::string& description) {
    if (gpu_ptr) {
        // For now, simulate GPU memory deallocation with host memory free
        // In real implementation, this would call cudaFree(gpu_ptr)
        free(gpu_ptr);
        std::cout << "Freed GPU memory: " << description << std::endl;
    }
    return GPUAllocationStatus::SUCCESS;
}

// Cleanup GPU memory
GPUAllocationStatus GPUMemoryManager::cleanup() {
    if (!m_initialized) {
        return GPUAllocationStatus::SUCCESS;
    }
    
    GPUAllocationStatus status = GPUAllocationStatus::SUCCESS;
    
    // Free all allocated memory
    if (m_gpu_memory.dag_memory) {
        GPUAllocationStatus dag_status = freeGPUMemory(m_gpu_memory.dag_memory, "DAG");
        if (dag_status != GPUAllocationStatus::SUCCESS) status = dag_status;
        m_gpu_memory.dag_memory = nullptr;
        m_gpu_memory.dag_size = 0;
    }
    
    if (m_gpu_memory.job_blobs_memory) {
        GPUAllocationStatus job_status = freeGPUMemory(m_gpu_memory.job_blobs_memory, "Job Blobs");
        if (job_status != GPUAllocationStatus::SUCCESS) status = job_status;
        m_gpu_memory.job_blobs_memory = nullptr;
        m_gpu_memory.job_blobs_size = 0;
    }
    
    if (m_gpu_memory.results_memory) {
        GPUAllocationStatus result_status = freeGPUMemory(m_gpu_memory.results_memory, "Results");
        if (result_status != GPUAllocationStatus::SUCCESS) status = result_status;
        m_gpu_memory.results_memory = nullptr;
        m_gpu_memory.results_size = 0;
    }
    
    if (m_gpu_memory.cache_memory) {
        GPUAllocationStatus cache_status = freeGPUMemory(m_gpu_memory.cache_memory, "Cache");
        if (cache_status != GPUAllocationStatus::SUCCESS) status = cache_status;
        m_gpu_memory.cache_memory = nullptr;
        m_gpu_memory.cache_size = 0;
    }
    
    m_gpu_memory.allocated = false;
    m_initialized = false;
    
    std::cout << "GPU Memory Manager cleanup completed" << std::endl;
    return status;
}

// Get total allocated memory
size_t GPUMemoryManager::getTotalAllocatedMemory() const {
    return m_gpu_memory.dag_size + m_gpu_memory.job_blobs_size + 
           m_gpu_memory.results_size + m_gpu_memory.cache_size;
}

// Print memory statistics
void GPUMemoryManager::printMemoryStats() const {
    std::cout << "\n📊 GPU Memory Statistics" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Device ID: " << m_device_id << std::endl;
    std::cout << "Initialized: " << (m_initialized ? "Yes" : "No") << std::endl;
    std::cout << "Allocated: " << (m_gpu_memory.allocated ? "Yes" : "No") << std::endl;
    std::cout << "DAG Memory: " << m_gpu_memory.dag_size << " bytes" << std::endl;
    std::cout << "Job Blobs Memory: " << m_gpu_memory.job_blobs_size << " bytes" << std::endl;
    std::cout << "Results Memory: " << m_gpu_memory.results_size << " bytes" << std::endl;
    std::cout << "Cache Memory: " << m_gpu_memory.cache_size << " bytes" << std::endl;
    std::cout << "Total Allocated: " << getTotalAllocatedMemory() << " bytes" << std::endl;
    
    if (!m_last_error.empty()) {
        std::cout << "Last Error: " << m_last_error << std::endl;
    }
    std::cout << std::endl;
}

// Set error message
void GPUMemoryManager::setError(const std::string& error) {
    m_last_error = error;
    std::cerr << "❌ GPU Memory Manager Error: " << error << std::endl;
}

// Log allocation
void GPUMemoryManager::logAllocation(const std::string& description, size_t size) {
    std::cout << "✅ Allocated GPU memory: " << description << " (" << size << " bytes)" << std::endl;
}

// Log transfer
void GPUMemoryManager::logTransfer(const std::string& description, size_t size) {
    std::cout << "🔄 GPU transfer: " << description << " (" << size << " bytes)" << std::endl;
}

} // namespace BMAD