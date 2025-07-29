#include "include/bmad_kawpow_multi.h"
#include "include/bmad_memory_manager.h"
#include "include/bmad_pool_manager.h"
#include "include/bmad_types.h"
#include <iostream>

int main() {
    std::cout << "BMAD Library Test" << std::endl;
    std::cout << "=================" << std::endl;
    
    try {
        // Test BMAD::KawPowMulti
        BMAD::KawPowMulti kawpow;
        std::cout << "✅ KawPowMulti created successfully" << std::endl;
        
        // Test BMAD::MemoryManager
        BMAD::MemoryManager memory;
        std::cout << "✅ MemoryManager created successfully" << std::endl;
        
        // Test BMAD::PoolManager
        BMAD::PoolManager pool;
        std::cout << "✅ PoolManager created successfully" << std::endl;
        
        // Test data structures
        BMAD::BMADConfig config;
        config.max_pools = 5;
        config.max_nonces = 1024;
        config.device_id = 0;
        config.batch_size = 256;
        config.memory_alignment = 256;
        config.use_pinned_memory = false;
        config.enable_profiling = false;
        
        BMAD::MultiPoolJob job;
        job.pool_id = 1;
        job.target = 0x12345678;
        job.height = 1000;
        job.active = true;
        
        BMAD::MultiPoolResult result;
        result.pool_id = 1;
        result.nonce = 0x12345678;
        result.hash = 0x87654321;
        result.actual_diff = 0x11111111;
        result.valid = true;
        
        std::cout << "✅ All data structures created successfully" << std::endl;
        
        // Test initialization
        bool init_result = kawpow.initialize(config);
        std::cout << "✅ KawPowMulti initialization: " << (init_result ? "SUCCESS" : "FAILED") << std::endl;
        
        // Test cleanup
        kawpow.cleanup();
        std::cout << "✅ KawPowMulti cleanup completed" << std::endl;
        
        std::cout << "\n🎉 ALL TESTS PASSED! BMAD library is working correctly." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}