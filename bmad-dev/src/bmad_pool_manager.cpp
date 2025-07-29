#include "bmad_pool_manager.h"
#include "bmad_types.h"
#include <iostream>
#include <algorithm>

namespace BMAD {

PoolManager::PoolManager() 
    : m_max_pools(10), m_active_pools(0) {
}

PoolManager::~PoolManager() {
    cleanup();
}

bool PoolManager::initialize(uint32_t max_pools) {
    m_max_pools = max_pools;
    m_pools.clear();
    m_pools.reserve(max_pools);
    
    std::cout << "Pool manager initialized with max pools: " << max_pools << std::endl;
    return true;
}

bool PoolManager::addPool(uint32_t pool_id, const std::string& name) {
    if (pool_id >= m_max_pools) {
        std::cerr << "Pool ID " << pool_id << " exceeds max pools " << m_max_pools << std::endl;
        return false;
    }
    
    // Check if pool already exists
    for (const auto& pool : m_pools) {
        if (pool.pool_id == pool_id) {
            std::cerr << "Pool " << pool_id << " already exists" << std::endl;
            return false;
        }
    }
    
    PoolInfo pool;
    pool.pool_id = pool_id;
    pool.name = name;
    pool.active = false;
    pool.connected = false;
    pool.job_count = 0;
    pool.share_count = 0;
    
    m_pools.push_back(pool);
    std::cout << "Added pool: " << name << " (ID: " << pool_id << ")" << std::endl;
    
    return true;
}

bool PoolManager::removePool(uint32_t pool_id) {
    auto it = std::find_if(m_pools.begin(), m_pools.end(),
                           [pool_id](const PoolInfo& pool) { return pool.pool_id == pool_id; });
    
    if (it == m_pools.end()) {
        std::cerr << "Pool " << pool_id << " not found" << std::endl;
        return false;
    }
    
    std::cout << "Removed pool: " << it->name << " (ID: " << pool_id << ")" << std::endl;
    m_pools.erase(it);
    
    return true;
}

bool PoolManager::setPoolActive(uint32_t pool_id, bool active) {
    for (auto& pool : m_pools) {
        if (pool.pool_id == pool_id) {
            pool.active = active;
            if (active) {
                m_active_pools++;
            } else {
                m_active_pools--;
            }
            std::cout << "Pool " << pool.name << " " << (active ? "activated" : "deactivated") << std::endl;
            return true;
        }
    }
    
    std::cerr << "Pool " << pool_id << " not found" << std::endl;
    return false;
}

bool PoolManager::setPoolConnected(uint32_t pool_id, bool connected) {
    for (auto& pool : m_pools) {
        if (pool.pool_id == pool_id) {
            pool.connected = connected;
            std::cout << "Pool " << pool.name << " " << (connected ? "connected" : "disconnected") << std::endl;
            return true;
        }
    }
    
    std::cerr << "Pool " << pool_id << " not found" << std::endl;
    return false;
}

bool PoolManager::updatePoolJob(uint32_t pool_id, const MultiPoolJob& job) {
    for (auto& pool : m_pools) {
        if (pool.pool_id == pool_id) {
            pool.current_job = job;
            pool.job_count++;
            std::cout << "Updated job for pool " << pool.name << " (job #" << pool.job_count << ")" << std::endl;
            return true;
        }
    }
    
    std::cerr << "Pool " << pool_id << " not found" << std::endl;
    return false;
}

bool PoolManager::submitShare(uint32_t pool_id, const MultiPoolResult& result) {
    for (auto& pool : m_pools) {
        if (pool.pool_id == pool_id) {
            pool.share_count++;
            std::cout << "Submitted share to pool " << pool.name << " (share #" << pool.share_count << ")" << std::endl;
            return true;
        }
    }
    
    std::cerr << "Pool " << pool_id << " not found" << std::endl;
    return false;
}

std::vector<MultiPoolJob> PoolManager::getActiveJobs() const {
    std::vector<MultiPoolJob> jobs;
    
    for (const auto& pool : m_pools) {
        if (pool.active && pool.connected && pool.current_job.active) {
            jobs.push_back(pool.current_job);
        }
    }
    
    return jobs;
}

std::vector<PoolInfo> PoolManager::getActivePools() const {
    std::vector<PoolInfo> active_pools;
    
    for (const auto& pool : m_pools) {
        if (pool.active && pool.connected) {
            active_pools.push_back(pool);
        }
    }
    
    return active_pools;
}

PoolInfo* PoolManager::getPool(uint32_t pool_id) {
    for (auto& pool : m_pools) {
        if (pool.pool_id == pool_id) {
            return &pool;
        }
    }
    
    return nullptr;
}

uint32_t PoolManager::getActivePoolCount() const {
    return m_active_pools;
}

uint32_t PoolManager::getTotalPoolCount() const {
    return static_cast<uint32_t>(m_pools.size());
}

void PoolManager::printStats() const {
    std::cout << "\nPool Manager Statistics:" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << "Total pools: " << getTotalPoolCount() << std::endl;
    std::cout << "Active pools: " << getActivePoolCount() << std::endl;
    
    for (const auto& pool : m_pools) {
        std::cout << "Pool " << pool.pool_id << " (" << pool.name << "): ";
        std::cout << (pool.active ? "active" : "inactive") << ", ";
        std::cout << (pool.connected ? "connected" : "disconnected") << ", ";
        std::cout << "jobs: " << pool.job_count << ", shares: " << pool.share_count << std::endl;
    }
}

void PoolManager::cleanup() {
    m_pools.clear();
    m_active_pools = 0;
    std::cout << "Pool manager cleaned up" << std::endl;
}

} // namespace BMAD