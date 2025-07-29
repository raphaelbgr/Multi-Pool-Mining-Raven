#include <iostream>
#include <vector>
#include "include/bmad_types.h"
#include "include/bmad_kawpow_multi.h"

using namespace BMAD;

int main() {
    std::cout << "BMAD Multi-Pool KawPow Test" << std::endl;
    std::cout << "===========================" << std::endl;
    
    // Test configuration
    BMADConfig config;
    config.max_pools = 5;
    config.batch_size = 1024;
    config.memory_alignment = 4096;
    config.use_pinned_memory = true;
    config.enable_profiling = false;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Max pools: " << config.max_pools << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Memory alignment: " << config.memory_alignment << std::endl;
    std::cout << "  Pinned memory: " << (config.use_pinned_memory ? "enabled" : "disabled") << std::endl;
    
    // Test multi-pool job creation
    std::vector<MultiPoolJob> test_jobs;
    for (uint32_t i = 0; i < config.max_pools; ++i) {
        MultiPoolJob job;
        job.pool_id = i;
        job.height = 1000000 + i;
        job.target = 0x00000000FFFFFFFFULL;
        job.active = true;
        
        // Fill blob with test data
        for (int j = 0; j < 40; ++j) {
            job.blob[j] = (uint8_t)(i + j);
        }
        
        test_jobs.push_back(job);
    }
    
    std::cout << "\nCreated " << test_jobs.size() << " test jobs" << std::endl;
    
    // Test memory alignment
    size_t aligned_size = MemoryManager::alignSize(1024, 4096);
    std::cout << "Memory alignment test: 1024 -> " << aligned_size << std::endl;
    
    // Test result structure
    MultiPoolResult test_result;
    test_result.nonce = 12345;
    test_result.pool_id = 0;
    test_result.hash = 0x123456789ABCDEF0ULL;
    test_result.valid = true;
    
    std::cout << "\nTest result:" << std::endl;
    std::cout << "  Nonce: " << test_result.nonce << std::endl;
    std::cout << "  Pool ID: " << test_result.pool_id << std::endl;
    std::cout << "  Hash: 0x" << std::hex << test_result.hash << std::dec << std::endl;
    std::cout << "  Valid: " << (test_result.valid ? "yes" : "no") << std::endl;
    
    std::cout << "\nBMAD setup test completed successfully!" << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "1. Run 'build.bat' to compile the library" << std::endl;
    std::cout << "2. Integrate with XMRig for multi-pool mining" << std::endl;
    std::cout << "3. Configure your RavenCoin pools" << std::endl;
    
    return 0;
}