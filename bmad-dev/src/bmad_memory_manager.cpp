#include "bmad_memory_manager.h"
#include "bmad_types.h"
#include <iostream>
#include <cstring>
#include <algorithm>

namespace BMAD {

MemoryManager::MemoryManager() 
    : m_device_id(0), m_initialized(false), m_pinned_job_blobs(nullptr), 
      m_pinned_targets(nullptr), m_pinned_results(nullptr) {
    memset(&m_memory, 0, sizeof(BMADMemory));
}

MemoryManager::~MemoryManager() {
    cleanup();
}

bool MemoryManager::initialize(const BMADConfig& config, uint32_t device_id) {
    m_device_id = device_id;
    m_config = config;
    
    // Initialize memory structure
    m_memory.max_pools = config.max_pools;
    m_memory.num_pools = 0;
    
    // Allocate memory for job blobs (host memory for now)
    size_t blob_size = config.max_pools * 40; // 40 bytes per job blob
    m_memory.job_blobs = malloc(blob_size);
    if (!m_memory.job_blobs) {
        std::cerr << "Failed to allocate job blobs memory" << std::endl;
        return false;
    }
    
    // Allocate memory for targets
    size_t target_size = config.max_pools * sizeof(uint64_t);
    m_memory.targets = malloc(target_size);
    if (!m_memory.targets) {
        std::cerr << "Failed to allocate targets memory" << std::endl;
        free(m_memory.job_blobs);
        m_memory.job_blobs = nullptr;
        return false;
    }
    
    // Allocate results buffer
    size_t results_size = config.batch_size * sizeof(uint32_t);
    m_memory.results = malloc(results_size);
    if (!m_memory.results) {
        std::cerr << "Failed to allocate results memory" << std::endl;
        free(m_memory.job_blobs);
        free(m_memory.targets);
        m_memory.job_blobs = nullptr;
        m_memory.targets = nullptr;
        return false;
    }
    
    m_initialized = true;
    std::cout << "Memory manager initialized for device " << device_id << std::endl;
    return true;
}

bool MemoryManager::prepareDAG(const uint64_t height) {
    if (!m_initialized) {
        return false;
    }
    
    // Free existing DAG and cache if they exist
    if (m_memory.dag) {
        free(m_memory.dag);
        m_memory.dag = nullptr;
    }
    
    if (m_memory.cache) {
        free(m_memory.cache);
        m_memory.cache = nullptr;
    }
    
    // Allocate cache memory (placeholder)
    size_t cache_size = 1024 * 1024; // 1MB placeholder
    m_memory.cache = malloc(cache_size);
    if (!m_memory.cache) {
        std::cerr << "Failed to allocate cache memory" << std::endl;
        return false;
    }
    m_memory.cache_size = cache_size;
    
    // Allocate DAG memory (placeholder)
    size_t dag_size = 4 * 1024 * 1024; // 4MB placeholder
    m_memory.dag = malloc(dag_size);
    if (!m_memory.dag) {
        std::cerr << "Failed to allocate DAG memory" << std::endl;
        free(m_memory.cache);
        m_memory.cache = nullptr;
        return false;
    }
    m_memory.dag_size = dag_size;
    
    // Initialize with placeholder data
    memset(m_memory.cache, 0xAA, cache_size);
    memset(m_memory.dag, 0xBB, dag_size);
    
    std::cout << "DAG prepared for height " << height << std::endl;
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
    
    // Copy job blobs to memory
    uint8_t* job_blobs = static_cast<uint8_t*>(m_memory.job_blobs);
    uint64_t* targets = static_cast<uint64_t*>(m_memory.targets);
    
    for (size_t i = 0; i < jobs.size(); i++) {
        memcpy(job_blobs + (i * 40), jobs[i].blob, 40);
        targets[i] = jobs[i].target;
    }
    
    m_memory.num_pools = static_cast<uint32_t>(jobs.size());
    std::cout << "Set " << jobs.size() << " jobs in memory manager" << std::endl;
    return true;
}

bool MemoryManager::allocateDAG(size_t dag_size) {
    if (m_memory.dag) {
        free(m_memory.dag);
    }
    
    m_memory.dag = malloc(dag_size);
    if (!m_memory.dag) {
        return false;
    }
    
    m_memory.dag_size = dag_size;
    return true;
}

bool MemoryManager::allocateCache(size_t cache_size) {
    if (m_memory.cache) {
        free(m_memory.cache);
    }
    
    m_memory.cache = malloc(cache_size);
    if (!m_memory.cache) {
        return false;
    }
    
    m_memory.cache_size = cache_size;
    return true;
}

bool MemoryManager::allocateJobMemory(uint32_t num_pools) {
    // Already allocated in initialize()
    return true;
}

bool MemoryManager::allocateResultsMemory(uint32_t max_results) {
    if (m_memory.results) {
        free(m_memory.results);
    }
    
    size_t results_size = max_results * sizeof(uint32_t);
    m_memory.results = malloc(results_size);
    if (!m_memory.results) {
        return false;
    }
    
    return true;
}

bool MemoryManager::copyDAGToDevice(const void* host_dag, size_t dag_size) {
    if (!m_memory.dag) {
        if (!allocateDAG(dag_size)) {
            return false;
        }
    }
    
    memcpy(m_memory.dag, host_dag, dag_size);
    return true;
}

bool MemoryManager::copyCacheToDevice(const void* host_cache, size_t cache_size) {
    if (!m_memory.cache) {
        if (!allocateCache(cache_size)) {
            return false;
        }
    }
    
    memcpy(m_memory.cache, host_cache, cache_size);
    return true;
}

bool MemoryManager::copyJobsToDevice(const std::vector<MultiPoolJob>& jobs) {
    return setJobs(jobs);
}

void MemoryManager::cleanup() {
    if (m_memory.dag) {
        free(m_memory.dag);
        m_memory.dag = nullptr;
    }
    
    if (m_memory.cache) {
        free(m_memory.cache);
        m_memory.cache = nullptr;
    }
    
    if (m_memory.job_blobs) {
        free(m_memory.job_blobs);
        m_memory.job_blobs = nullptr;
    }
    
    if (m_memory.targets) {
        free(m_memory.targets);
        m_memory.targets = nullptr;
    }
    
    if (m_memory.results) {
        free(m_memory.results);
        m_memory.results = nullptr;
    }
    
    m_initialized = false;
}

size_t MemoryManager::alignSize(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void* MemoryManager::alignPointer(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
}

bool MemoryManager::allocateDeviceMemory(void** ptr, size_t size) {
    *ptr = malloc(size);
    return *ptr != nullptr;
}

bool MemoryManager::allocatePinnedMemory(void** ptr, size_t size) {
    *ptr = malloc(size);
    return *ptr != nullptr;
}

void MemoryManager::freeDeviceMemory(void* ptr) {
    free(ptr);
}

void MemoryManager::freePinnedMemory(void* ptr) {
    free(ptr);
}

} // namespace BMAD