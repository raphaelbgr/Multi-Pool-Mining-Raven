#ifndef BMAD_KAWPOW_OPTIMIZED_H
#define BMAD_KAWPOW_OPTIMIZED_H

#include "bmad_kawpow_algorithm.h"
#include <cstdint>
#include <vector>

namespace BMAD {

// Optimized multi-pool constants
#define OPTIMIZED_BLOCK_SIZE 256
#define OPTIMIZED_GRID_SIZE 1024
#define SHARED_MEMORY_SIZE 16384
#define MAX_POOLS_PER_BLOCK 8
#define OPTIMIZED_NONCES_PER_THREAD 4

// Optimized thread block configuration
struct OptimizedBlockConfig {
    uint32_t block_size;
    uint32_t grid_size;
    uint32_t shared_memory_size;
    uint32_t max_pools_per_block;
    uint32_t nonces_per_thread;
};

// Optimized multi-pool job structure
struct OptimizedMultiPoolJob {
    uint8_t* job_blobs[MAX_POOLS_PER_BLOCK];
    uint64_t targets[MAX_POOLS_PER_BLOCK];
    uint32_t pool_count;
    uint32_t start_nonce;
    uint32_t nonce_count;
};

// Optimized result structure
struct OptimizedMultiPoolResult {
    uint32_t* results;
    uint32_t* nonces;
    uint32_t* result_count;
    uint32_t max_results;
};

// Optimized KawPow kernel class
class KawPowOptimized {
public:
    // Initialize optimized kernel
    static bool initialize(const OptimizedBlockConfig& config);
    
    // Calculate optimized multi-pool hash
    static bool calculateOptimizedMultiPoolHash(
        const OptimizedMultiPoolJob& job,
        OptimizedMultiPoolResult& result,
        const uint8_t* dag,
        size_t dag_size
    );
    
    // Optimized Keccak-f[800] with shared memory
    static void keccakf800Optimized(uint32_t state[25], uint32_t* shared_memory);
    
    // Optimized ProgPow mix with shared memory
    static uint32_t progpowMixOptimized(
        uint32_t state[PROGPOW_REGS], 
        uint32_t nonce, 
        uint32_t lane_id,
        uint32_t* shared_memory
    );
    
    // Optimized DAG access with caching
    static uint32_t dagAccessOptimized(
        const uint8_t* dag, 
        size_t dag_size, 
        uint32_t index,
        uint32_t* shared_cache
    );
    
    // Batch processing for multiple pools
    static bool processBatch(
        const std::vector<OptimizedMultiPoolJob>& jobs,
        std::vector<OptimizedMultiPoolResult>& results,
        const uint8_t* dag,
        size_t dag_size
    );
    
    // Performance monitoring
    static void startPerformanceMonitoring();
    static void endPerformanceMonitoring();
    static void printPerformanceStats();
    
    // Cleanup
    static void cleanup();

private:
    // Internal optimization functions
    static void optimizeMemoryLayout(uint32_t* shared_memory, uint32_t size);
    static void synchronizeThreads();
    static uint32_t calculateOptimalBlockSize(uint32_t pool_count);
    static uint32_t calculateOptimalGridSize(uint32_t nonce_count);
    
    // Performance tracking
    static uint64_t m_start_time;
    static uint64_t m_end_time;
    static uint32_t m_processed_hashes;
    static uint32_t m_found_shares;
};

} // namespace BMAD

#endif // BMAD_KAWPOW_OPTIMIZED_H