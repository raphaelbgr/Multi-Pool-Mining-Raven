#ifndef BMAD_KAWPOW_ALGORITHM_H
#define BMAD_KAWPOW_ALGORITHM_H

#include <cstdint>
#include <cstring>

namespace BMAD {

// KawPow constants
#define PROGPOW_LANES 32
#define PROGPOW_REGS 32
#define PROGPOW_CACHE_WORDS 2048
#define PROGPOW_DAG_LOADS 4
#define PROGPOW_CACHE_BYTES (PROGPOW_CACHE_WORDS * 4)
#define PROGPOW_CN_MEMORY 2097152
#define PROGPOW_CN_MEMORY_BYTES (PROGPOW_CN_MEMORY * 8)
#define PROGPOW_CN_DAG_LOADS 4
#define PROGPOW_CN_DAG_LOADS_BYTES (PROGPOW_CN_DAG_LOADS * 8)

// Keccak-f[800] constants
extern const uint32_t keccakf_rndc[24];
extern const uint32_t keccakf_rotc[24];
extern const uint32_t keccakf_piln[24];

// KawPow algorithm functions
class KawPowAlgorithm {
public:
    // Initialize KawPow algorithm
    static bool initialize();
    
    // Calculate KawPow hash for a single job
    static uint64_t calculateHash(
        const uint8_t* job_blob,
        uint32_t nonce,
        const uint8_t* dag,
        size_t dag_size
    );
    
    // Calculate KawPow hash for multiple pools simultaneously
    static bool calculateMultiPoolHash(
        uint8_t* job_blobs[],
        uint64_t targets[],
        uint32_t pool_count,
        uint32_t* results,
        uint32_t* nonces,
        uint32_t* result_count,
        uint32_t start_nonce,
        uint32_t nonce_count,
        uint8_t* dag,
        size_t dag_size
    );
    
    // Keccak-f[800] function
    static void keccakf800(uint32_t state[25]);
    
    // ProgPow mix function
    static uint32_t progpowMix(uint32_t state[PROGPOW_REGS], uint32_t nonce, uint32_t lane_id);
    
    // DAG access function
    static uint32_t dagAccess(const uint8_t* dag, size_t dag_size, uint32_t index);
    
    // Cleanup
    static void cleanup();

    // Public helper functions for optimized kernel
    static void initializeState(uint32_t state[25], const uint8_t* job_blob, uint32_t nonce);
    static void applyRavencoinConstraints(uint32_t state[25]);
    static uint64_t finalizeHash(uint32_t state[25]);
};

} // namespace BMAD

#endif // BMAD_KAWPOW_ALGORITHM_H