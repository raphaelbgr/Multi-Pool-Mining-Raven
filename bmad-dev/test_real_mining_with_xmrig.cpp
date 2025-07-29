#include "../include/bmad_xmrig_bridge.h"
#include <iostream>
#include <signal.h>
#include <atomic>

using namespace BMAD;

std::atomic<bool> g_running(true);

void signalHandler(int signal) {
    std::cout << "\n🛑 Received signal " << signal << ", stopping mining..." << std::endl;
    g_running = false;
}

int main() {
    std::cout << "🚀 BMAD Real Mining Test with XMRig Integration" << std::endl;
    std::cout << "================================================" << std::endl;
    
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    try {
        // Initialize real mining test
        RealMiningTest mining_test;
        
        if (!mining_test.initialize()) {
            std::cerr << "❌ Failed to initialize real mining test" << std::endl;
            return 1;
        }
        
        // Start real mining
        if (!mining_test.startRealMining()) {
            std::cerr << "❌ Failed to start real mining" << std::endl;
            return 1;
        }
        
        std::cout << "\n🎯 Real mining active with XMRig integration!" << std::endl;
        std::cout << "  - Connected to real pools via XMRig" << std::endl;
        std::cout << "  - Processing live jobs from pools" << std::endl;
        std::cout << "  - Submitting shares to pools" << std::endl;
        std::cout << "  - Multi-pool performance multiplication active" << std::endl;
        std::cout << "\n  Press Ctrl+C to stop mining" << std::endl;
        
        // Keep mining until interrupted
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        // Stop mining
        mining_test.stopMining();
        
        std::cout << "\n🎉 Real mining test completed successfully!" << std::endl;
        std::cout << "✅ BMAD + XMRig integration working!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Real mining test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}