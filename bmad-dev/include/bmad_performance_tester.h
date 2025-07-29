#ifndef BMAD_PERFORMANCE_TESTER_H
#define BMAD_PERFORMANCE_TESTER_H

#include "bmad_types.h"
#include "bmad_gpu_memory_manager.h"
#include "bmad_kawpow_multi.h"
#include <vector>
#include <chrono>
#include <string>
#include <memory>

namespace BMAD {

// Performance test configuration
struct PerformanceTestConfig {
    uint32_t min_pools;
    uint32_t max_pools;
    uint32_t nonces_per_test;
    uint32_t iterations_per_test;
    uint32_t warmup_iterations;
    bool use_gpu_memory;
    bool use_optimized_kernel;
    bool validate_results;
    
    PerformanceTestConfig() : min_pools(1), max_pools(5), nonces_per_test(10000),
                             iterations_per_test(10), warmup_iterations(3),
                             use_gpu_memory(true), use_optimized_kernel(true),
                             validate_results(true) {}
};

// Performance test result
struct PerformanceTestResult {
    uint32_t pool_count;
    double hashrate_hps;
    double duration_ms;
    uint32_t shares_found;
    size_t memory_used_bytes;
    double efficiency_percent;
    std::string test_name;
    
    PerformanceTestResult() : pool_count(0), hashrate_hps(0.0), duration_ms(0.0),
                             shares_found(0), memory_used_bytes(0), efficiency_percent(0.0) {}
};

// Performance comparison result
struct PerformanceComparison {
    std::vector<PerformanceTestResult> single_pool_results;
    std::vector<PerformanceTestResult> multi_pool_results;
    double performance_multiplier;
    double memory_efficiency;
    double scalability_score;
    
    PerformanceComparison() : performance_multiplier(0.0), memory_efficiency(0.0), scalability_score(0.0) {}
};

// Forward declaration for KawPowOptimized
class KawPowOptimized;

// Performance tester for BMAD framework
class PerformanceTester {
public:
    PerformanceTester();
    ~PerformanceTester();
    
    // Initialize performance tester
    bool initialize(const PerformanceTestConfig& config);
    
    // Run comprehensive performance tests
    PerformanceComparison runComprehensiveTests();
    
    // Individual test methods
    PerformanceTestResult testSinglePoolMining(uint32_t pool_count);
    PerformanceTestResult testMultiPoolMining(uint32_t pool_count);
    PerformanceTestResult testOptimizedKernel(uint32_t pool_count);
    PerformanceTestResult testGPUMemoryPerformance(uint32_t pool_count);
    
    // Benchmark methods
    std::vector<PerformanceTestResult> benchmarkPoolScaling();
    std::vector<PerformanceTestResult> benchmarkMemoryUsage();
    std::vector<PerformanceTestResult> benchmarkKernelOptimization();
    
    // Analysis methods
    PerformanceComparison analyzeResults(const std::vector<PerformanceTestResult>& results);
    void printPerformanceReport(const PerformanceComparison& comparison);
    void exportResultsToCSV(const std::vector<PerformanceTestResult>& results, const std::string& filename);
    
    // Utility methods
    std::vector<MultiPoolJob> generateTestJobs(uint32_t pool_count);
    void warmupSystem();
    double calculateHashrate(uint32_t nonces, double duration_ms);
    double calculateEfficiency(const PerformanceTestResult& result);

private:
    PerformanceTestConfig m_config;
    std::unique_ptr<GPUMemoryManager> m_gpu_manager;
    std::unique_ptr<KawPowMulti> m_kawpow_multi;
    std::unique_ptr<KawPowOptimized> m_kawpow_optimized;
    
    // Test data
    std::vector<uint8_t> m_test_dag;
    std::vector<MultiPoolJob> m_test_jobs;
    
    // Helper methods
    void generateTestData();
    PerformanceTestResult runSingleTest(uint32_t pool_count, bool use_multi_pool);
    void measureMemoryUsage(size_t& memory_used);
    void validateTestResults(const PerformanceTestResult& result);
    std::string formatHashrate(double hashrate);
    std::string formatDuration(double duration_ms);
};

} // namespace BMAD

#endif // BMAD_PERFORMANCE_TESTER_H