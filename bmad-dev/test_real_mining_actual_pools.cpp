#include "../include/bmad_kawpow_multi.h"
#include "../include/bmad_pool_connector.h"
#include "../include/bmad_share_converter.h"
#include "../include/bmad_memory_manager.h"
#include "../include/bmad_pool_manager.h"
#include "../include/bmad_types.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <atomic>

using namespace BMAD;

std::atomic<bool> g_running(true);

void signalHandler(int signal) {
    std::cout << "\n🛑 Received signal " << signal << ", stopping mining..." << std::endl;
    g_running = false;
}

int main() {
    std::cout << "🚀 BMAD Real Mining Test with ACTUAL Pools" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    try {
        // Initialize BMAD components
        std::cout << "🔧 Initializing BMAD components..." << std::endl;
        
        auto kawpow_multi = std::make_unique<KawPowMulti>();
        auto memory_manager = std::make_unique<MemoryManager>();
        auto pool_manager = std::make_unique<PoolManager>();
        auto pool_connector = std::make_unique<PoolConnector>();
        auto share_converter = std::make_unique<ShareConverter>();
        
        // Configure BMAD
        BMADConfig config;
        config.device_id = 0;
        config.max_pools = 5;
        config.max_nonces = 100000;
        
        if (!kawpow_multi->initialize(config)) {
            std::cerr << "❌ Failed to initialize KawPow Multi" << std::endl;
            return 1;
        }
        
        if (!memory_manager->initialize(config, 0)) {
            std::cerr << "❌ Failed to initialize Memory Manager" << std::endl;
            return 1;
        }
        
        if (!pool_manager->initialize(5)) {
            std::cerr << "❌ Failed to initialize Pool Manager" << std::endl;
            return 1;
        }
        
        if (!pool_connector->initialize()) {
            std::cerr << "❌ Failed to initialize Pool Connector" << std::endl;
            return 1;
        }
        
        std::cout << "✅ BMAD components initialized successfully" << std::endl;
        
        // Add real pool configurations
        std::cout << "🔌 Adding real pool configurations..." << std::endl;
        
        // Pool 1: 2Miners
        pool_connector->addPool(1, "rvn.2miners.com", "6060", 
                               "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU", "x", "stratum1");
        
        // Pool 2: Ravenminer
        pool_connector->addPool(2, "stratum.ravenminer.com", "3838", 
                               "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU.Yoda", "x", "stratum1");
        
        // Pool 3: WoolyPooly
        pool_connector->addPool(3, "pool.br.woolypooly.com", "55555", 
                               "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU.Obiwan", "x", "stratum2");
        
        // Pool 4: HeroMiners
        pool_connector->addPool(4, "br.ravencoin.herominers.com", "1140", 
                               "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU.Ahsoka", "x", "stratum2");
        
        // Pool 5: NanoPool
        pool_connector->addPool(5, "rvn-us-east1.nanopool.org", "10400", 
                               "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU.Luke", "x", "stratum1");
        
        std::cout << "✅ Pool configurations added" << std::endl;
        
        // Connect to pools
        std::cout << "🔌 Connecting to real pools..." << std::endl;
        
        if (!pool_connector->connectToPools()) {
            std::cerr << "❌ Failed to connect to pools" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Connected to pools successfully" << std::endl;
        
        // Start mining loop
        std::cout << "⛏️  Starting real mining with actual pools..." << std::endl;
        std::cout << "  Press Ctrl+C to stop mining" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        uint32_t total_shares = 0;
        uint32_t valid_shares = 0;
        
        while (g_running) {
            try {
                // Get current jobs from pools
                auto jobs = pool_connector->getCurrentJobs();
                
                if (!jobs.empty()) {
                    std::cout << "📥 Received " << jobs.size() << " jobs from pools" << std::endl;
                    
                    // Set jobs for multi-pool mining
                    kawpow_multi->setJobs(jobs);
                    
                    // Mine with BMAD
                    uint32_t rescount = 0, resnonce = 0;
                    kawpow_multi->mine(0, &rescount, &resnonce);
                    
                    if (rescount > 0) {
                        total_shares += rescount;
                        std::cout << "🎯 Found " << rescount << " shares (nonce: " << resnonce << ")" << std::endl;
                        
                        // Process each share
                        for (uint32_t i = 0; i < rescount; ++i) {
                            uint32_t share_nonce = resnonce + i;
                            
                            // Convert share to XMRig format
                            xmrig::JobResult xmrig_result;
                            if (share_converter->convertBMADShareToXMRig(jobs[0], share_nonce, 1, xmrig_result)) {
                                
                                // Validate share
                                if (share_converter->validateShare(xmrig_result, jobs[0].target)) {
                                    valid_shares++;
                                    
                                    // Submit share to pool
                                    if (pool_connector->submitShare(1, xmrig_result.jobId, share_nonce, xmrig_result.result)) {
                                        std::cout << "📤 Share submitted successfully to pool 1" << std::endl;
                                    } else {
                                        std::cout << "❌ Failed to submit share to pool 1" << std::endl;
                                    }
                                } else {
                                    std::cout << "❌ Share validation failed" << std::endl;
                                }
                            } else {
                                std::cout << "❌ Failed to convert share" << std::endl;
                            }
                        }
                    }
                }
                
                // Print statistics every 10 seconds
                static auto last_stats_time = std::chrono::high_resolution_clock::now();
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time);
                
                if (elapsed.count() >= 10) {
                    auto mining_time = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
                    
                    std::cout << "\n📊 Mining Statistics:" << std::endl;
                    std::cout << "  Duration: " << mining_time.count() << " seconds" << std::endl;
                    std::cout << "  Total shares: " << total_shares << std::endl;
                    std::cout << "  Valid shares: " << valid_shares << std::endl;
                    std::cout << "  Success rate: " << (total_shares > 0 ? (valid_shares * 100.0 / total_shares) : 0) << "%" << std::endl;
                    
                    // Check pool connections
                    for (uint32_t i = 1; i <= 5; ++i) {
                        if (pool_connector->isPoolConnected(i)) {
                            uint32_t submitted, accepted, rejected;
                            pool_connector->getPoolStatistics(i, submitted, accepted, rejected);
                            std::cout << "  Pool " << i << ": " << submitted << " submitted, " 
                                     << accepted << " accepted, " << rejected << " rejected" << std::endl;
                        } else {
                            std::cout << "  Pool " << i << ": DISCONNECTED" << std::endl;
                        }
                    }
                    
                    last_stats_time = now;
                }
                
                // Small delay
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
            } catch (const std::exception& e) {
                std::cerr << "❌ Exception in mining loop: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        
        // Final statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "\n🎉 Real mining test completed!" << std::endl;
        std::cout << "📊 Final Statistics:" << std::endl;
        std::cout << "  Total duration: " << total_duration.count() << " seconds" << std::endl;
        std::cout << "  Total shares found: " << total_shares << std::endl;
        std::cout << "  Valid shares: " << valid_shares << std::endl;
        std::cout << "  Success rate: " << (total_shares > 0 ? (valid_shares * 100.0 / total_shares) : 0) << "%" << std::endl;
        
        // Cleanup
        std::cout << "🧹 Cleaning up..." << std::endl;
        pool_connector->disconnectFromPools();
        kawpow_multi->cleanup();
        
        std::cout << "✅ Real mining test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Real mining test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}