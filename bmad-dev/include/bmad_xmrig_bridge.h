#ifndef BMAD_XMRIG_BRIDGE_H
#define BMAD_XMRIG_BRIDGE_H

#include "bmad_types.h"
#include "bmad_kawpow_multi.h"
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <chrono>

// Forward declarations for XMRig components
namespace xmrig {
    class IClient;
    class Job;
    class JobResult;
    class Pool;
    class IStrategy;
    class IStrategyListener;
}

namespace BMAD {

// Bridge between BMAD and XMRig for real mining
class XMRigBridge {
public:
    XMRigBridge();
    ~XMRigBridge();
    
    // Initialize bridge with XMRig components
    bool initialize(xmrig::IStrategy* strategy, xmrig::IStrategyListener* listener);
    
    // Connect BMAD to XMRig's job system
    bool connectJobSystem();
    
    // Connect BMAD to XMRig's share submission system
    bool connectShareSubmission();
    
    // Start real mining with XMRig integration
    bool startRealMining();
    
    // Stop mining
    void stopMining();
    
    // Get mining statistics
    uint32_t getSharesFound() const { return m_shares_found; }
    uint32_t getActivePools() const { return m_active_pools; }
    double getHashrate() const { return m_current_hashrate; }
    
    // Cleanup
    void cleanup();

private:
    // XMRig components
    xmrig::IStrategy* m_strategy;
    xmrig::IStrategyListener* m_listener;
    
    // BMAD components
    std::unique_ptr<KawPowMulti> m_kawpow_multi;
    
    // Mining state
    std::atomic<bool> m_mining_active;
    std::atomic<uint32_t> m_shares_found;
    std::atomic<uint32_t> m_active_pools;
    std::atomic<double> m_current_hashrate;
    
    // Callbacks for XMRig integration
    std::function<void(const xmrig::Job&)> m_job_callback;
    std::function<void(const xmrig::JobResult&)> m_share_callback;
    
    // Internal methods
    void onJobReceived(const xmrig::Job& job);
    void onShareFound(const xmrig::JobResult& result);
    void processJobsWithBMAD(const std::vector<xmrig::Job>& jobs);
    void submitSharesToXMRig(const std::vector<xmrig::JobResult>& shares);
    
    // Mining loop
    void miningLoop();
    std::thread m_mining_thread;
};

// Real mining test with XMRig integration
class RealMiningTest {
public:
    RealMiningTest();
    ~RealMiningTest();
    
    // Initialize with real XMRig integration
    bool initialize();
    
    // Start real mining with actual pools
    bool startRealMining();
    
    // Stop mining
    void stopMining();
    
    // Get statistics
    void printStatistics() const;
    
    // Cleanup
    void cleanup();

private:
    std::unique_ptr<XMRigBridge> m_bridge;
    std::chrono::high_resolution_clock::time_point m_start_time;
    bool m_mining_active;
    
    // Statistics
    uint32_t m_total_shares;
    uint32_t m_accepted_shares;
    uint32_t m_rejected_shares;
    double m_total_hashrate;
};

} // namespace BMAD

#endif // BMAD_XMRIG_BRIDGE_H