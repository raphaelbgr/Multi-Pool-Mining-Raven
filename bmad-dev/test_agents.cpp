#include <iostream>
#include <vector>
#include <memory>
#include "bmad_agent_manager.h"
#include "bmad_types.h"

using namespace BMAD;

int main() {
    std::cout << "BMAD Multi-Pool KawPow Agents Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Create agent manager
    std::unique_ptr<AgentManager> agent_manager = std::make_unique<AgentManager>();
    
    // Initialize with default config
    BMADConfig config;
    config.max_pools = 5;
    config.batch_size = 1024;
    config.memory_alignment = 4096;
    config.use_pinned_memory = true;
    config.enable_profiling = false;
    
    if (!agent_manager->initialize(config, 0)) {
        std::cout << "ERROR: Failed to initialize agent manager" << std::endl;
        return 1;
    }
    
    std::cout << "Agent manager initialized successfully" << std::endl;
    
    // Add test agents
    std::vector<std::string> pool_names = {
        "2miners", "RavenMiner", "WoolyPooly", "HeroMiners", "NanoPool"
    };
    
    for (size_t i = 0; i < pool_names.size(); ++i) {
        if (agent_manager->addAgent(pool_names[i], static_cast<uint32_t>(i), static_cast<uint32_t>(i + 1))) {
            std::cout << "Added agent: " << pool_names[i] << " (pool_id: " << i << ")" << std::endl;
        } else {
            std::cout << "Failed to add agent: " << pool_names[i] << std::endl;
        }
    }
    
    // Test agent retrieval
    std::cout << "\nTesting agent retrieval..." << std::endl;
    for (uint32_t i = 0; i < 5; ++i) {
        PoolAgent* agent = agent_manager->getAgent(i);
        if (agent) {
            std::cout << "Found agent: " << agent->name << " (enabled: " << (agent->enabled ? "yes" : "no") << ")" << std::endl;
        } else {
            std::cout << "Agent " << i << " not found" << std::endl;
        }
    }
    
    // Test active agents
    std::cout << "\nActive agents:" << std::endl;
    auto active_agents = agent_manager->getActiveAgents();
    for (const auto& agent : active_agents) {
        std::cout << "- " << agent->name << " (priority: " << agent->priority << ")" << std::endl;
    }
    
    // Test agent statistics
    std::cout << "\nAgent statistics:" << std::endl;
    std::cout << "Total agents: " << agent_manager->getTotalAgentCount() << std::endl;
    std::cout << "Active agents: " << agent_manager->getActiveAgentCount() << std::endl;
    
    // Test agent enable/disable
    std::cout << "\nTesting agent enable/disable..." << std::endl;
    if (agent_manager->setAgentEnabled(2, false)) {
        std::cout << "Disabled agent 2 (WoolyPooly)" << std::endl;
    }
    
    active_agents = agent_manager->getActiveAgents();
    std::cout << "Active agents after disabling one: " << active_agents.size() << std::endl;
    
    // Test job update
    std::cout << "\nTesting job update..." << std::endl;
    MultiPoolJob test_job;
    memset(test_job.blob, 0xAA, sizeof(test_job.blob));
    test_job.target = 1000000;
    test_job.height = 3952812;
    test_job.pool_id = 0;
    test_job.active = true;
    
    if (agent_manager->updateAgentJob(0, test_job)) {
        std::cout << "Updated job for agent 0" << std::endl;
    } else {
        std::cout << "Failed to update job for agent 0" << std::endl;
    }
    
    // Test mining (simulation)
    std::cout << "\nTesting mining simulation..." << std::endl;
    std::vector<MultiPoolResult> results;
    if (agent_manager->mine(0, 1000, results)) {
        std::cout << "Mining simulation completed" << std::endl;
        std::cout << "Results count: " << results.size() << std::endl;
    } else {
        std::cout << "Mining simulation failed" << std::endl;
    }
    
    // Cleanup
    agent_manager->cleanup();
    std::cout << "\nCleanup completed" << std::endl;
    
    std::cout << "\nBMAD Agents Test Completed Successfully!" << std::endl;
    return 0;
}