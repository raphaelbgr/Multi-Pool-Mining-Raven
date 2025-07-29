#include "../include/bmad_pool_connector.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

// Windows socket includes
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#endif

namespace BMAD {

PoolConnector::PoolConnector() 
    : m_running(false) {
    std::cout << "🔌 Pool Connector created" << std::endl;
    
    // Initialize Winsock on Windows
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "❌ Failed to initialize Winsock" << std::endl;
    }
#endif
}

PoolConnector::~PoolConnector() {
    cleanup();
    
#ifdef _WIN32
    WSACleanup();
#endif
    
    std::cout << "🔌 Pool Connector destroyed" << std::endl;
}

bool PoolConnector::initialize() {
    std::cout << "🔌 Initializing Pool Connector..." << std::endl;
    m_running = true;
    std::cout << "✅ Pool Connector initialized" << std::endl;
    return true;
}

bool PoolConnector::addPool(uint32_t pool_id, const std::string& url, const std::string& port, 
                           const std::string& user, const std::string& pass, const std::string& adapter) {
    std::cout << "🔌 Adding pool " << pool_id << ": " << url << ":" << port << std::endl;
    
    auto pool = std::make_unique<PoolConnection>();
    pool->pool_id = pool_id;
    pool->url = url;
    pool->user = user;
    pool->pass = pass;
    pool->adapter = adapter;
    pool->status = PoolConnectionStatus::DISCONNECTED;
    pool->socket_fd = -1;
    pool->connected = false;
    pool->authenticated = false;
    pool->subscribed = false;
    pool->has_job = false;
    pool->shares_submitted = 0;
    pool->shares_accepted = 0;
    pool->shares_rejected = 0;
    pool->last_share_time = 0;
    
    // Parse port
    try {
        pool->port = std::stoi(port);
    } catch (const std::exception& e) {
        std::cerr << "❌ Invalid port number: " << port << std::endl;
        return false;
    }
    
    {
        std::lock_guard<std::mutex> lock(m_pools_mutex);
        m_pools.push_back(std::move(pool));
    }
    
    std::cout << "✅ Pool " << pool_id << " added successfully" << std::endl;
    return true;
}

bool PoolConnector::connectToPools() {
    std::cout << "🔌 Connecting to pools..." << std::endl;
    
    std::lock_guard<std::mutex> lock(m_pools_mutex);
    
    for (auto& pool : m_pools) {
        if (!connectToPool(pool.get())) {
            std::cerr << "❌ Failed to connect to pool " << pool->pool_id << std::endl;
            continue;
        }
        
        if (!authenticatePool(pool.get())) {
            std::cerr << "❌ Failed to authenticate with pool " << pool->pool_id << std::endl;
            continue;
        }
        
        if (!subscribeToPool(pool.get())) {
            std::cerr << "❌ Failed to subscribe to pool " << pool->pool_id << std::endl;
            continue;
        }
        
        std::cout << "✅ Pool " << pool->pool_id << " connected and ready" << std::endl;
    }
    
    return true;
}

void PoolConnector::disconnectFromPools() {
    std::cout << "🔌 Disconnecting from pools..." << std::endl;
    
    std::lock_guard<std::mutex> lock(m_pools_mutex);
    
    for (auto& pool : m_pools) {
        if (pool->socket_fd != -1) {
#ifdef _WIN32
            closesocket(pool->socket_fd);
#else
            close(pool->socket_fd);
#endif
            pool->socket_fd = -1;
        }
        pool->connected = false;
        pool->authenticated = false;
        pool->subscribed = false;
    }
    
    std::cout << "✅ Disconnected from all pools" << std::endl;
}

bool PoolConnector::submitShare(uint32_t pool_id, const std::string& job_id, uint32_t nonce, 
                               const std::string& result) {
    std::lock_guard<std::mutex> lock(m_pools_mutex);
    
    for (auto& pool : m_pools) {
        if (pool->pool_id == pool_id) {
            return submitShareToPool(pool.get(), job_id, nonce, result);
        }
    }
    
    std::cerr << "❌ Pool " << pool_id << " not found" << std::endl;
    return false;
}

std::vector<MultiPoolJob> PoolConnector::getCurrentJobs() {
    std::vector<MultiPoolJob> jobs;
    std::lock_guard<std::mutex> lock(m_pools_mutex);
    
    for (auto& pool : m_pools) {
        if (pool->has_job && pool->connected) {
            jobs.push_back(pool->current_job);
        }
    }
    
    return jobs;
}

bool PoolConnector::isPoolConnected(uint32_t pool_id) const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_pools_mutex));
    
    for (const auto& pool : m_pools) {
        if (pool->pool_id == pool_id) {
            return pool->connected;
        }
    }
    
    return false;
}

void PoolConnector::getPoolStatistics(uint32_t pool_id, uint32_t& shares_submitted, 
                                     uint32_t& shares_accepted, uint32_t& shares_rejected) {
    std::lock_guard<std::mutex> lock(m_pools_mutex);
    
    for (const auto& pool : m_pools) {
        if (pool->pool_id == pool_id) {
            shares_submitted = pool->shares_submitted;
            shares_accepted = pool->shares_accepted;
            shares_rejected = pool->shares_rejected;
            return;
        }
    }
    
    shares_submitted = 0;
    shares_accepted = 0;
    shares_rejected = 0;
}

void PoolConnector::cleanup() {
    std::cout << "🧹 Cleaning up Pool Connector..." << std::endl;
    disconnectFromPools();
    std::cout << "✅ Pool Connector cleanup completed" << std::endl;
}

// Private methods

bool PoolConnector::connectToPool(PoolConnection* pool) {
    std::cout << "  🔌 Connecting to " << pool->url << ":" << pool->port << std::endl;
    
    pool->socket_fd = createSocket(pool->url, pool->port);
    if (pool->socket_fd == -1) {
        std::cerr << "    ❌ Failed to create socket" << std::endl;
        return false;
    }
    
    pool->status = PoolConnectionStatus::CONNECTED;
    pool->connected = true;
    
    std::cout << "    ✅ Connected successfully" << std::endl;
    return true;
}

bool PoolConnector::authenticatePool(PoolConnection* pool) {
    std::cout << "  🔐 Authenticating with pool " << pool->pool_id << std::endl;
    
    std::string auth_request = createStratumAuthorize(pool->user, pool->pass);
    if (!sendData(pool->socket_fd, auth_request)) {
        std::cerr << "    ❌ Failed to send authentication request" << std::endl;
        return false;
    }
    
    std::string response = receiveData(pool->socket_fd);
    if (response.empty()) {
        std::cerr << "    ❌ No response to authentication" << std::endl;
        return false;
    }
    
    bool success;
    std::string error;
    if (!parseStratumResponse(response, success, error)) {
        std::cerr << "    ❌ Failed to parse authentication response" << std::endl;
        return false;
    }
    
    if (!success) {
        std::cerr << "    ❌ Authentication failed: " << error << std::endl;
        return false;
    }
    
    pool->status = PoolConnectionStatus::AUTHENTICATED;
    pool->authenticated = true;
    
    std::cout << "    ✅ Authentication successful" << std::endl;
    return true;
}

bool PoolConnector::subscribeToPool(PoolConnection* pool) {
    std::cout << "  📡 Subscribing to pool " << pool->pool_id << std::endl;
    
    std::string subscribe_request = createStratumSubscribe();
    if (!sendData(pool->socket_fd, subscribe_request)) {
        std::cerr << "    ❌ Failed to send subscription request" << std::endl;
        return false;
    }
    
    std::string response = receiveData(pool->socket_fd);
    if (response.empty()) {
        std::cerr << "    ❌ No response to subscription" << std::endl;
        return false;
    }
    
    bool success;
    std::string error;
    if (!parseStratumResponse(response, success, error)) {
        std::cerr << "    ❌ Failed to parse subscription response" << std::endl;
        return false;
    }
    
    if (!success) {
        std::cerr << "    ❌ Subscription failed: " << error << std::endl;
        return false;
    }
    
    pool->status = PoolConnectionStatus::SUBSCRIBED;
    pool->subscribed = true;
    
    std::cout << "    ✅ Subscription successful" << std::endl;
    return true;
}

bool PoolConnector::submitShareToPool(PoolConnection* pool, const std::string& job_id, 
                                     uint32_t nonce, const std::string& result) {
    std::string submit_request = createStratumSubmit(pool->user, job_id, nonce, result);
    if (!sendData(pool->socket_fd, submit_request)) {
        std::cerr << "    ❌ Failed to send share submission" << std::endl;
        return false;
    }
    
    std::string response = receiveData(pool->socket_fd);
    if (response.empty()) {
        std::cerr << "    ❌ No response to share submission" << std::endl;
        return false;
    }
    
    bool success;
    std::string error;
    if (!parseStratumResponse(response, success, error)) {
        std::cerr << "    ❌ Failed to parse share submission response" << std::endl;
        return false;
    }
    
    pool->shares_submitted++;
    if (success) {
        pool->shares_accepted++;
        std::cout << "    ✅ Share accepted by pool " << pool->pool_id << std::endl;
    } else {
        pool->shares_rejected++;
        std::cerr << "    ❌ Share rejected by pool " << pool->pool_id << ": " << error << std::endl;
    }
    
    return success;
}

int PoolConnector::createSocket(const std::string& host, int port) {
#ifdef _WIN32
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        return -1;
    }
#else
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        return -1;
    }
#endif
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    // Resolve hostname
    struct hostent* he = gethostbyname(host.c_str());
    if (he == nullptr) {
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return -1;
    }
    
    server_addr.sin_addr = *((struct in_addr*)he->h_addr);
    
    // Connect
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) != 0) {
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return -1;
    }
    
    return sock;
}

bool PoolConnector::sendData(int socket_fd, const std::string& data) {
    if (socket_fd == -1) {
        return false;
    }
    
    const char* buffer = data.c_str();
    int total_sent = 0;
    int data_length = static_cast<int>(data.length());
    
    while (total_sent < data_length) {
        int sent = send(socket_fd, buffer + total_sent, data_length - total_sent, 0);
        if (sent == -1) {
            return false;
        }
        total_sent += sent;
    }
    
    return true;
}

std::string PoolConnector::receiveData(int socket_fd, int timeout_ms) {
    if (socket_fd == -1) {
        return "";
    }
    
    // Set timeout
#ifdef _WIN32
    DWORD timeout = timeout_ms;
    setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
#else
    struct timeval timeout;
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
    
    char buffer[4096];
    std::string response;
    
    int received = recv(socket_fd, buffer, sizeof(buffer) - 1, 0);
    if (received > 0) {
        buffer[received] = '\0';
        response = std::string(buffer);
    }
    
    return response;
}

std::string PoolConnector::createStratumSubscribe() {
    return "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"BMAD/1.0.0\", null]}\n";
}

std::string PoolConnector::createStratumAuthorize(const std::string& user, const std::string& pass) {
    return "{\"id\": 2, \"method\": \"mining.authorize\", \"params\": [\"" + user + "\", \"" + pass + "\"]}\n";
}

std::string PoolConnector::createStratumSubmit(const std::string& user, const std::string& job_id, 
                                              uint32_t nonce, const std::string& result) {
    std::ostringstream oss;
    oss << "{\"id\": 4, \"method\": \"mining.submit\", \"params\": [\"" << user << "\", \"" 
        << job_id << "\", \"" << std::hex << nonce << "\", \"" << result << "\"]}\n";
    return oss.str();
}

bool PoolConnector::parseStratumResponse(const std::string& response, bool& success, std::string& error) {
    // Simple JSON parsing for Stratum responses
    // In a real implementation, you'd use a proper JSON library
    
    if (response.find("\"result\":true") != std::string::npos) {
        success = true;
        error = "";
        return true;
    }
    
    if (response.find("\"result\":false") != std::string::npos) {
        success = false;
        // Extract error message if available
        size_t error_start = response.find("\"error\":");
        if (error_start != std::string::npos) {
            size_t error_end = response.find("\"", error_start + 9);
            if (error_end != std::string::npos) {
                error = response.substr(error_start + 9, error_end - error_start - 9);
            }
        }
        return true;
    }
    
    // Unknown response format
    success = false;
    error = "Unknown response format";
    return false;
}

bool PoolConnector::parseStratumJob(const std::string& response, MultiPoolJob& job) {
    // Parse job notification from pool
    // This is a simplified implementation
    
    // Extract job parameters from Stratum notification
    // In a real implementation, you'd parse the full JSON structure
    
    // For now, create a simulated job
    job.pool_id = 1; // Will be set by caller
    job.height = 1000;
    job.target = 0x12345678;
    
    // Fill job blob with some data
    for (size_t i = 0; i < sizeof(job.blob); ++i) {
        job.blob[i] = static_cast<uint8_t>(i % 256);
    }
    
    return true;
}

} // namespace BMAD