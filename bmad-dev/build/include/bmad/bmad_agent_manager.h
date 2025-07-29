#ifndef BMAD_AGENT_MANAGER_H
#define BMAD_AGENT_MANAGER_H

#include "bmad_kawpow_multi.h"
#include "bmad_types.h"
#include <vector>
#include <memory>
#include <string>

namespace BMAD {

// Individual pool agent
struct PoolAgent {
    std::string name;
    uint32_t pool_id;
    bool enabled;
    uint32_t priority;
    MultiPoolJob current_job;
    bool has_job;
    
    PoolAgent(const std::string& n, uint32_t id, uint32_t p) 
        : name(n), pool_id(id), enabled(true), priority(p), has_job(false) {}
};

// Agent manager for handling multiple pool agents
class AgentManager {
public:
    AgentManager();
    ~AgentManager();
    
    // Initialize agent manager
    bool initialize(const BMADConfig& config, uint32_t device_id);
    
    // Add pool agent
    bool addAgent(const std::string& name, uint32_t pool_id, uint32_t priority = 0);
    
    // Remove pool agent
    bool removeAgent(uint32_t pool_id);
    
    // Update agent job
    bool updateAgentJob(uint32_t pool_id, const MultiPoolJob& job);
    
    // Get active agents
    std::vector<PoolAgent*> getActiveAgents();
    
    // Mine with all active agents
    bool mine(uint32_t start_nonce, uint32_t num_nonces, 
              std::vector<MultiPoolResult>& results);
    
    // Get agent statistics
    uint32_t getActiveAgentCount() const { return m_active_agents; }
    uint32_t getTotalAgentCount() const { return m_agents.size(); }
    
    // Enable/disable agent
    bool setAgentEnabled(uint32_t pool_id, bool enabled);
    
    // Get agent by pool ID
    PoolAgent* getAgent(uint32_t pool_id);
    
    // Cleanup
    void cleanup();

private:
    std::unique_ptr<KawPowMulti> m_bmad_miner;
    std::vector<std::unique_ptr<PoolAgent>> m_agents;
    BMADConfig m_config;
    uint32_t m_device_id;
    uint32_t m_active_agents;
    
    // Internal methods
    bool updateBMADJobs();
    void sortAgentsByPriority();
    bool validateAgent(const PoolAgent* agent);
};

} // namespace BMAD

#endif // BMAD_AGENT_MANAGER_H