#include "../include/bmad_performance_tester.h"
#include <iostream>
#include <iomanip> // Required for std::fixed and std::setprecision

using namespace BMAD;

int main() {
    std::cout << "🧪 BMAD Performance Tester" << std::endl;
    std::cout << "===========================" << std::endl;
    
    try {
        // Initialize performance tester
        PerformanceTestConfig config;
        config.min_pools = 1;
        config.max_pools = 5;
        config.nonces_per_test = 10000;
        config.iterations_per_test = 5;
        config.warmup_iterations = 2;
        config.use_gpu_memory = true;
        config.use_optimized_kernel = true;
        config.validate_results = true;
        
        PerformanceTester tester;
        
        if (!tester.initialize(config)) {
            std::cerr << "Failed to initialize Performance Tester" << std::endl;
            return 1;
        }
        
        // Run comprehensive performance tests
        std::cout << "\n🚀 Starting comprehensive performance tests..." << std::endl;
        auto comparison = tester.runComprehensiveTests();
        
        // Print summary
        std::cout << "\n🎉 PERFORMANCE TESTING COMPLETED!" << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "✅ Pool scaling benchmark completed" << std::endl;
        std::cout << "✅ Memory usage benchmark completed" << std::endl;
        std::cout << "✅ Kernel optimization benchmark completed" << std::endl;
        std::cout << "✅ Performance analysis completed" << std::endl;
        std::cout << "✅ Results exported to CSV" << std::endl;
        
        std::cout << "\n📊 Final Results:" << std::endl;
        std::cout << "  Performance Multiplier: " << std::fixed << std::setprecision(2) 
                  << comparison.performance_multiplier << "x" << std::endl;
        std::cout << "  Memory Efficiency: " << std::fixed << std::setprecision(2) 
                  << comparison.memory_efficiency << " H/s per KB" << std::endl;
        std::cout << "  Scalability Score: " << std::fixed << std::setprecision(2) 
                  << comparison.scalability_score << std::endl;
        
        std::cout << "\n🚀 BMAD Performance Testing ready for production!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Performance testing failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}