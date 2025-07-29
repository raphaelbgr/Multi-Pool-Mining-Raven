#include "bmad_memory_manager.h"
#include "bmad_types.h"
#include <iostream>
#include <cuda_runtime.h>

namespace BMAD {

MemoryManager::MemoryManager() 
    : m_device_id(0), m_initialized(false) {
    memset(&m_memory, 0, sizeof(BMADMemory));
}

MemoryManager::~MemoryManager() {
    cleanup();
}

bool MemoryManager::initialize(const BMADConfig& config, uint32_t device_id) {
    m_device_id = device_id;
    
    // Set CUDA device
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Initialize memory structure
    m_memory.max_pools = config.max_pools;
    m_memory.num_pools = 0;
    
    // Allocate memory for job blobs
    size_t blob_size = config.max_pools * 40; // 40 bytes per job blob
    error = cudaMalloc(&m_memory.job_blobs, blob_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate job blobs memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate memory for targets
    size_t target_size = config.max_pools * sizeof(uint64_t);
    error = cudaMalloc(&m_memory.targets, target_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate targets memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate results buffer
    size_t results_size = config.batch_size * sizeof(uint32_t);
    error = cudaMalloc(&m_memory.results, results_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate results memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    m_initialized = true;
    std::cout << "Memory manager initialized for device " << device_id << std::endl;
    return true;
}

bool MemoryManager::prepareDAG(const void* cache, size_t cache_size, size_t dag_size, uint32_t height) {
    if (!m_initialized) {
        return false;
    }
    
    // Free existing DAG and cache if they exist
    if (m_memory.dag) {
        cudaFree(m_memory.dag);
        m_memory.dag = nullptr;
    }
    
    if (m_memory.cache) {
        cudaFree(m_memory.cache);
        m_memory.cache = nullptr;
    }
    
    // Allocate cache memory
    cudaError_t error = cudaMalloc(&m_memory.cache, cache_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate cache memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy cache to device
    error = cudaMemcpy(m_memory.cache, cache, cache_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy cache to device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate DAG memory
    error = cudaMalloc(&m_memory.dag, dag_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate DAG memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Calculate DAG (this would be the actual KawPow DAG calculation)
    // For now, we'll just copy some data as a placeholder
    error = cudaMemcpy(m_memory.dag, m_memory.cache, std::min(cache_size, dag_size), cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to calculate DAG: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    m_memory.cache_size = cache_size;
    m_memory.dag_size = dag_size;
    
    std::cout << "DAG prepared: cache_size=" << cache_size << ", dag_size=" << dag_size << std::endl;
    return true;
}

bool MemoryManager::setJobs(const std::vector<MultiPoolJob>& jobs) {
    if (!m_initialized) {
        return false;
    }
    
    if (jobs.size() > m_memory.max_pools) {
        std::cerr << "Too many jobs: " << jobs.size() << " > " << m_memory.max_pools << std::endl;
        return false;
    }
    
    // Copy job blobs to device
    std::vector<uint8_t> blob_data;
    std::vector<uint64_t> target_data;
    
    for (const auto& job : jobs) {
        // Copy blob
        blob_data.insert(blob_data.end(), job.blob, job.blob + 40);
        target_data.push_back(job.target);
    }
    
    // Copy blobs to device
    cudaError_t error = cudaMemcpy(m_memory.job_blobs, blob_data.data(), 
                                   blob_data.size(), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy job blobs: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy targets to device
    error = cudaMemcpy(m_memory.targets, target_data.data(), 
                       target_data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy targets: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    m_memory.num_pools = static_cast<uint32_t>(jobs.size());
    
    std::cout << "Set " << jobs.size() << " jobs" << std::endl;
    return true;
}

bool MemoryManager::updateJob(const MultiPoolJob& job) {
    if (!m_initialized) {
        return false;
    }
    
    if (job.pool_id >= m_memory.max_pools) {
        return false;
    }
    
    // Update specific job blob
    size_t offset = job.pool_id * 40;
    cudaError_t error = cudaMemcpy(static_cast<uint8_t*>(m_memory.job_blobs) + offset,
                                   job.blob, 40, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to update job blob: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Update target
    offset = job.pool_id * sizeof(uint64_t);
    error = cudaMemcpy(static_cast<uint8_t*>(m_memory.targets) + offset,
                       &job.target, sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to update target: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

void MemoryManager::cleanup() {
    if (m_memory.dag) {
        cudaFree(m_memory.dag);
        m_memory.dag = nullptr;
    }
    
    if (m_memory.cache) {
        cudaFree(m_memory.cache);
        m_memory.cache = nullptr;
    }
    
    if (m_memory.job_blobs) {
        cudaFree(m_memory.job_blobs);
        m_memory.job_blobs = nullptr;
    }
    
    if (m_memory.targets) {
        cudaFree(m_memory.targets);
        m_memory.targets = nullptr;
    }
    
    if (m_memory.results) {
        cudaFree(m_memory.results);
        m_memory.results = nullptr;
    }
    
    m_initialized = false;
    std::cout << "Memory manager cleaned up" << std::endl;
}

const BMADMemory& MemoryManager::getMemory() const {
    return m_memory;
}

} // namespace BMAD