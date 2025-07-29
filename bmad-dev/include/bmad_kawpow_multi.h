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
    bool initialize(const BMADConfig& config);
    
    // Prepare DAG and cache for mining
    bool prepare(const uint64_t height);
    
    // Set multi-pool jobs
    bool setJobs(const std::vector<MultiPoolJob>& jobs);
    
    // Mine with multiple pools simultaneously
    bool mine(uint32_t start_nonce, uint32_t* rescount, uint32_t* resnonce);
    
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
    
    // Pool management
    uint32_t getActivePoolCount();
    uint32_t getTotalPoolCount();
    uint32_t getSkippedHashes();
    void printStats();

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
    
    // Internal methods
    bool validateJob(const MultiPoolJob& job);
    bool updateJob(const MultiPoolJob& job);
    void resetStatistics();
};

} // namespace BMAD

#endif // BMAD_KAWPOW_MULTI_H