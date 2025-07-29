#ifndef BMAD_POOL_MANAGER_H
#define BMAD_POOL_MANAGER_H

#include "bmad_types.h"
#include <vector>
#include <string>

namespace BMAD {

// Pool information structure
struct PoolInfo {
    uint32_t pool_id;
    std::string name;
    bool active;
    bool connected;
    uint32_t job_count;
    uint32_t share_count;
    MultiPoolJob current_job;
    
    PoolInfo() : pool_id(0), active(false), connected(false), job_count(0), share_count(0) {}
};

// Pool manager for handling multiple pool connections
class PoolManager {
public:
    PoolManager();
    ~PoolManager();
    
    // Initialize pool manager
    bool initialize(uint32_t max_pools = 10);
    
    // Pool management
    bool addPool(uint32_t pool_id, const std::string& name);
    bool removePool(uint32_t pool_id);
    bool setPoolActive(uint32_t pool_id, bool active);
    bool setPoolConnected(uint32_t pool_id, bool connected);
    
    // Job and share management
    bool updatePoolJob(uint32_t pool_id, const MultiPoolJob& job);
    bool submitShare(uint32_t pool_id, const MultiPoolResult& result);
    
    // Get active jobs and pools
    std::vector<MultiPoolJob> getActiveJobs() const;
    std::vector<PoolInfo> getActivePools() const;
    
    // Get specific pool
    PoolInfo* getPool(uint32_t pool_id);
    
    // Statistics
    uint32_t getActivePoolCount() const;
    uint32_t getTotalPoolCount() const;
    void printStats() const;
    
    // Cleanup
    void cleanup();

private:
    std::vector<PoolInfo> m_pools;
    uint32_t m_max_pools;
    uint32_t m_active_pools;
};

} // namespace BMAD

#endif // BMAD_POOL_MANAGER_H