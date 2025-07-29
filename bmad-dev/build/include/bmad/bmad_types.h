#ifndef BMAD_TYPES_H
#define BMAD_TYPES_H

#include <cstdint>
#include <vector>
#include <memory>

namespace BMAD {

// Core data types for KawPow
typedef struct {
    uint32_t uint32s[32 / sizeof(uint32_t)];
} hash32_t;

typedef struct {
    uint64_t uint64s[8];
} hash64_t;

// Multi-pool job structure
struct MultiPoolJob {
    uint8_t blob[40];           // Job blob for each pool
    uint64_t target;            // Target difficulty for each pool
    uint32_t height;            // Block height
    uint32_t pool_id;           // Pool identifier
    bool active;                // Whether this pool job is active
};

// BMAD memory management structure
struct BMADMemory {
    void* dag;                  // DAG memory
    void* cache;                // Cache memory
    void* job_blobs;            // Array of job blobs
    void* targets;              // Array of targets
    void* results;              // Results buffer
    size_t dag_size;            // DAG size
    size_t cache_size;          // Cache size
    uint32_t num_pools;         // Number of active pools
    uint32_t max_pools;         // Maximum number of pools
};

// BMAD context for multi-pool mining
struct BMADContext {
    BMADMemory memory;          // Memory management
    std::vector<MultiPoolJob> jobs;  // Pool jobs
    uint32_t device_id;         // CUDA device ID
    uint32_t blocks;            // CUDA blocks
    uint32_t threads;           // CUDA threads
    uint32_t intensity;         // Mining intensity
    bool initialized;            // Whether context is initialized
};

// Result structure for multi-pool shares
struct MultiPoolResult {
    uint32_t nonce;             // Found nonce
    uint32_t pool_id;           // Pool where share was found
    uint64_t hash;              // Hash value
    bool valid;                 // Whether share is valid
};

// BMAD configuration
struct BMADConfig {
    uint32_t max_pools;         // Maximum number of pools
    uint32_t batch_size;        // Batch size for processing
    uint32_t memory_alignment;  // Memory alignment
    bool use_pinned_memory;     // Use pinned memory
    bool enable_profiling;      // Enable CUDA profiling
};

} // namespace BMAD

#endif // BMAD_TYPES_H