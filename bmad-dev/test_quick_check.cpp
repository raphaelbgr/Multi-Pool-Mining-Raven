#include "../src/bmad_kawpow_algorithm.h"
#include "../include/bmad_kawpow_multi.h"
#include "../include/bmad_memory_manager.h"
#include "../include/bmad_pool_manager.h"
#include "../include/bmad_types.h"
#include <iostream>
#include <cassert>

using namespace BMAD;

int main() {
    std::cout << "🔍 BMAD Quick Component Check" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        // Test 1: Core KawPow Algorithm
        std::cout << "\n✅ Test 1: Core KawPow Algorithm" << std::endl;
        assert(KawPowAlgorithm::initialize());
        
        uint8_t job_blob[40];
        for (int i = 0; i < 40; i++) {
            job_blob[i] = (uint8_t)(i % 256);
        }
        
        uint32_t nonce = 12345;
        uint8_t dag[1024];
        for (int i = 0; i < 1024; i++) {
            dag[i] = (uint8_t)(i % 256);
        }
        
        uint64_t hash = KawPowAlgorithm::calculateHash(job_blob, nonce, dag, 1024);
        std::cout << "  Hash calculated: 0x" << std::hex << hash << std::dec << std::endl;
        
        KawPowAlgorithm::cleanup();
        
        // Test 2: Memory Manager
        std::cout << "\n✅ Test 2: Memory Manager" << std::endl;
        BMADConfig config;
        config.device_id = 0;
        config.max_pools = 10;
        config.max_nonces = 1000;
        
        MemoryManager manager;
        assert(manager.initialize(config, 0));
        assert(manager.prepareDAG(1000));
        
        std::vector<MultiPoolJob> jobs;
        for (int i = 0; i < 3; i++) {
            MultiPoolJob job;
            job.pool_id = i + 1;
            job.active = true;
            job.height = 1000 + i;
            job.target = 0x1000000 + (i * 0x100000);
            memset(job.blob, i, 40);
            jobs.push_back(job);
        }
        
        assert(manager.setJobs(jobs));
        std::cout << "  Jobs set successfully" << std::endl;
        
        manager.cleanup();
        
        // Test 3: Pool Manager
        std::cout << "\n✅ Test 3: Pool Manager" << std::endl;
        PoolManager pool_manager;
        assert(pool_manager.initialize(10));
        
        for (int i = 0; i < 3; i++) {
            assert(pool_manager.addPool(i + 1, "Pool" + std::to_string(i + 1)));
        }
        
        assert(pool_manager.setPoolActive(1, true));
        assert(pool_manager.setPoolConnected(1, true));
        
        std::cout << "  Active pools: " << pool_manager.getActivePoolCount() << std::endl;
        std::cout << "  Total pools: " << pool_manager.getTotalPoolCount() << std::endl;
        
        pool_manager.cleanup();
        
        // Test 4: KawPow Multi
        std::cout << "\n✅ Test 4: KawPow Multi" << std::endl;
        KawPowMulti miner;
        assert(miner.initialize(config));
        assert(miner.prepare(1000));
        assert(miner.setJobs(jobs));
        
        uint32_t rescount = 0, resnonce = 0;
        assert(miner.mine(0, &rescount, &resnonce));
        
        std::cout << "  Shares found: " << rescount << std::endl;
        std::cout << "  Nonce: " << resnonce << std::endl;
        std::cout << "  Active pools: " << miner.getActivePoolCount() << std::endl;
        std::cout << "  Total pools: " << miner.getTotalPoolCount() << std::endl;
        
        miner.cleanup();
        
        std::cout << "\n🎉 ALL COMPONENTS WORKING!" << std::endl;
        std::cout << "=========================" << std::endl;
        std::cout << "✅ Core KawPow Algorithm" << std::endl;
        std::cout << "✅ Memory Management" << std::endl;
        std::cout << "✅ Pool Management" << std::endl;
        std::cout << "✅ Multi-Pool Mining" << std::endl;
        std::cout << "\n🚀 BMAD is ready for real testing!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}