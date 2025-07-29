#ifndef BMAD_POOL_CONNECTOR_H
#define BMAD_POOL_CONNECTOR_H

#include "bmad_types.h"
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>

namespace BMAD {

// Pool connection status
enum class PoolConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    AUTHENTICATED,
    SUBSCRIBED,
    MINING
};

// Pool connection information
struct PoolConnection {
    uint32_t pool_id;
    std::string url;
    int port;
    std::string user;
    std::string pass;
    std::string adapter;
    PoolConnectionStatus status;
    int socket_fd;
    std::atomic<bool> connected;
    std::atomic<bool> authenticated;
    std::atomic<bool> subscribed;
    
    // Job information
    MultiPoolJob current_job;
    std::atomic<bool> has_job;
    
    // Statistics
    std::atomic<uint32_t> shares_submitted;
    std::atomic<uint32_t> shares_accepted;
    std::atomic<uint32_t> shares_rejected;
    std::atomic<uint64_t> last_share_time;
};

// Real pool connector for TCP connections
class PoolConnector {
public:
    PoolConnector();
    ~PoolConnector();
    
    // Initialize connector
    bool initialize();
    
    // Add pool configuration
    bool addPool(uint32_t pool_id, const std::string& url, const std::string& port,
                 const std::string& user, const std::string& pass, const std::string& adapter);
    
    // Connect to all pools
    bool connectToPools();
    
    // Disconnect from all pools
    void disconnectFromPools();
    
    // Submit share to specific pool
    bool submitShare(uint32_t pool_id, const std::string& job_id, uint32_t nonce, 
                    const std::string& result);
    
    // Get current jobs from all pools
    std::vector<MultiPoolJob> getCurrentJobs();
    
    // Check if pool is connected
    bool isPoolConnected(uint32_t pool_id) const;
    
    // Get pool statistics
    void getPoolStatistics(uint32_t pool_id, uint32_t& shares_submitted, 
                          uint32_t& shares_accepted, uint32_t& shares_rejected);
    
    // Cleanup
    void cleanup();

private:
    // Pool connections
    std::vector<std::unique_ptr<PoolConnection>> m_pools;
    std::mutex m_pools_mutex;
    
    // Connection threads
    std::vector<std::thread> m_connection_threads;
    std::atomic<bool> m_running;
    
    // Internal methods
    bool connectToPool(PoolConnection* pool);
    bool authenticatePool(PoolConnection* pool);
    bool subscribeToPool(PoolConnection* pool);
    bool submitShareToPool(PoolConnection* pool, const std::string& job_id, 
                          uint32_t nonce, const std::string& result);
    
    // TCP socket operations
    int createSocket(const std::string& host, int port);
    bool sendData(int socket_fd, const std::string& data);
    std::string receiveData(int socket_fd, int timeout_ms = 5000);
    
    // Stratum protocol methods
    std::string createStratumSubscribe();
    std::string createStratumAuthorize(const std::string& user, const std::string& pass);
    std::string createStratumSubmit(const std::string& user, const std::string& job_id, 
                                   uint32_t nonce, const std::string& result);
    
    // JSON parsing helpers
    bool parseStratumResponse(const std::string& response, bool& success, std::string& error);
    bool parseStratumJob(const std::string& response, MultiPoolJob& job);
    
    // Connection management
    void connectionLoop(PoolConnection* pool);
    void handlePoolDisconnection(PoolConnection* pool);
    void reconnectPool(PoolConnection* pool);
};

} // namespace BMAD

#endif // BMAD_POOL_CONNECTOR_H