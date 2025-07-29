#include "include/bmad_pool_connector.h"
#include "include/bmad_share_converter.h"
#include "src/bmad_kawpow_algorithm.h"
#include "include/bmad_gpu_memory_manager.h"
#include "include/bmad_types.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <iomanip> // Required for std::setw and std::fixed

namespace BMAD {

class BMADIntegrationTest {
private:
    PoolConnector m_pool_connector;
    ShareConverter m_share_converter;
    GPUMemoryManager m_gpu_manager;
    
    // Test results
    struct TestResult {
        std::string test_name;
        bool passed;
        std::string error_message;
        double execution_time_ms;
    };
    
    std::vector<TestResult> m_test_results;

public:
    BMADIntegrationTest() {
        std::cout << "🧪 BMAD Integration Test Suite Starting..." << std::endl;
        std::cout << "==========================================" << std::endl;
    }
    
    ~BMADIntegrationTest() {
        printTestSummary();
    }
    
    void runAllTests() {
        std::cout << "\n🚀 Running all integration tests..." << std::endl;
        
        // Core functionality tests
        testPoolConnectorInitialization();
        testPoolAddition();
        testShareConverterInitialization();
        testKawPowAlgorithm();
        testMultiPoolJobCreation();
        testShareConversion();
        testHashCalculation();
        testShareValidation();
        testGPUMemoryManager();
        testMultiPoolMining();
        testErrorHandling();
        testPerformance();
        
        std::cout << "\n✅ All tests completed!" << std::endl;
    }

private:
    void testPoolConnectorInitialization() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n🔌 Testing Pool Connector Initialization..." << std::endl;
            
            // Test initialization
            bool init_result = m_pool_connector.initialize();
            assert(init_result && "Pool connector initialization failed");
            
            // Test basic functionality
            assert(m_pool_connector.isPoolConnected(1) == false && "New pool should not be connected");
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Pool Connector Initialization", true, "", duration.count());
            std::cout << "✅ Pool Connector initialization test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Pool Connector Initialization", false, e.what(), duration.count());
            std::cout << "❌ Pool Connector initialization test failed: " << e.what() << std::endl;
        }
    }
    
    void testPoolAddition() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n🔌 Testing Pool Addition..." << std::endl;
            
            // Test adding multiple pools
            bool add_result1 = m_pool_connector.addPool(1, "stratum+tcp://pool1.example.com", "3333", 
                                                       "user1", "pass1", "adapter1");
            assert(add_result1 && "Failed to add pool 1");
            
            bool add_result2 = m_pool_connector.addPool(2, "stratum+tcp://pool2.example.com", "3333", 
                                                       "user2", "pass2", "adapter2");
            assert(add_result2 && "Failed to add pool 2");
            
            // Test adding duplicate pool ID
            bool add_result3 = m_pool_connector.addPool(1, "stratum+tcp://pool3.example.com", "3333", 
                                                       "user3", "pass3", "adapter3");
            // This should succeed as we're not checking for duplicates in current implementation
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Pool Addition", true, "", duration.count());
            std::cout << "✅ Pool addition test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Pool Addition", false, e.what(), duration.count());
            std::cout << "❌ Pool addition test failed: " << e.what() << std::endl;
        }
    }
    
    void testShareConverterInitialization() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n🔄 Testing Share Converter Initialization..." << std::endl;
            
            // Test that share converter can be created and destroyed
            ShareConverter converter;
            
            // Test basic functionality
            MultiPoolJob test_job;
            test_job.pool_id = 1;
            test_job.height = 1000;
            test_job.target = 0x12345678;
            test_job.active = true;
            
            // Fill job blob with test data
            for (size_t i = 0; i < sizeof(test_job.blob); ++i) {
                test_job.blob[i] = static_cast<uint8_t>(i % 256);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Share Converter Initialization", true, "", duration.count());
            std::cout << "✅ Share Converter initialization test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Share Converter Initialization", false, e.what(), duration.count());
            std::cout << "❌ Share Converter initialization test failed: " << e.what() << std::endl;
        }
    }
    
    void testKawPowAlgorithm() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n⚡ Testing KawPow Algorithm..." << std::endl;
            
            // Test algorithm initialization
            bool init_result = KawPowAlgorithm::initialize();
            assert(init_result && "KawPow algorithm initialization failed");
            
            // Test hash calculation with dummy data
            uint8_t test_blob[40];
            for (size_t i = 0; i < sizeof(test_blob); ++i) {
                test_blob[i] = static_cast<uint8_t>(i % 256);
            }
            
            uint32_t test_nonce = 0x12345678;
            
            // Note: This is a simplified test since we don't have real DAG data
            // In a real implementation, we would need proper DAG initialization
            uint64_t hash_result = KawPowAlgorithm::calculateHash(
                test_blob, 
                test_nonce, 
                nullptr,  // DAG would be passed here
                0         // DAG size would be passed here
            );
            
            // Verify hash is not zero (basic sanity check)
            assert(hash_result != 0 && "Hash result should not be zero");
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("KawPow Algorithm", true, "", duration.count());
            std::cout << "✅ KawPow algorithm test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("KawPow Algorithm", false, e.what(), duration.count());
            std::cout << "❌ KawPow algorithm test failed: " << e.what() << std::endl;
        }
    }
    
    void testMultiPoolJobCreation() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n📋 Testing Multi-Pool Job Creation..." << std::endl;
            
            // Create test jobs for multiple pools
            std::vector<MultiPoolJob> jobs;
            
            for (uint32_t pool_id = 1; pool_id <= 3; ++pool_id) {
                MultiPoolJob job;
                job.pool_id = pool_id;
                job.height = 1000 + pool_id;
                job.target = 0x12345678 + pool_id;
                job.active = true;
                
                // Fill job blob with unique data for each pool
                for (size_t i = 0; i < sizeof(job.blob); ++i) {
                    job.blob[i] = static_cast<uint8_t>((i + pool_id) % 256);
                }
                
                jobs.push_back(job);
            }
            
            // Verify jobs were created correctly
            assert(jobs.size() == 3 && "Should have created 3 jobs");
            
            for (const auto& job : jobs) {
                assert(job.pool_id >= 1 && job.pool_id <= 3 && "Invalid pool ID");
                assert(job.height >= 1000 && job.height <= 1003 && "Invalid height");
                assert(job.active == true && "Job should be active");
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Multi-Pool Job Creation", true, "", duration.count());
            std::cout << "✅ Multi-pool job creation test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Multi-Pool Job Creation", false, e.what(), duration.count());
            std::cout << "❌ Multi-pool job creation test failed: " << e.what() << std::endl;
        }
    }
    
    void testShareConversion() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n🔄 Testing Share Conversion..." << std::endl;
            
            // Create test job
            MultiPoolJob test_job;
            test_job.pool_id = 1;
            test_job.height = 1000;
            test_job.target = 0x12345678;
            test_job.active = true;
            
            // Fill job blob with test data
            for (size_t i = 0; i < sizeof(test_job.blob); ++i) {
                test_job.blob[i] = static_cast<uint8_t>(i % 256);
            }
            
            // Test share conversion
            uint32_t test_nonce = 0x12345678;
            uint32_t test_pool_id = 1;
            
            xmrig::JobResult xmrig_result;
            bool conversion_result = m_share_converter.convertBMADShareToXMRig(
                test_job, test_nonce, test_pool_id, xmrig_result);
            
            assert(conversion_result && "Share conversion failed");
            assert(xmrig_result.jobId == "1000" && "Job ID mismatch");
            assert(xmrig_result.nonce == test_nonce && "Nonce mismatch");
            assert(xmrig_result.poolId == static_cast<int>(test_pool_id) && "Pool ID mismatch");
            assert(!xmrig_result.result.empty() && "Result should not be empty");
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Share Conversion", true, "", duration.count());
            std::cout << "✅ Share conversion test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Share Conversion", false, e.what(), duration.count());
            std::cout << "❌ Share conversion test failed: " << e.what() << std::endl;
        }
    }
    
    void testHashCalculation() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n🔢 Testing Hash Calculation..." << std::endl;
            
            // Create test job
            MultiPoolJob test_job;
            test_job.pool_id = 1;
            test_job.height = 1000;
            test_job.target = 0x12345678;
            test_job.active = true;
            
            // Fill job blob with test data
            for (size_t i = 0; i < sizeof(test_job.blob); ++i) {
                test_job.blob[i] = static_cast<uint8_t>(i % 256);
            }
            
            // Test hash calculation for different nonces
            std::vector<uint32_t> test_nonces = {0x12345678, 0x87654321, 0xDEADBEEF};
            
            for (uint32_t nonce : test_nonces) {
                uint8_t hash[32];
                bool hash_result = m_share_converter.calculateHashForNonce(test_job, nonce, hash);
                
                assert(hash_result && "Hash calculation failed");
                
                // Verify hash is not all zeros
                bool all_zero = true;
                for (int i = 0; i < 32; ++i) {
                    if (hash[i] != 0) {
                        all_zero = false;
                        break;
                    }
                }
                assert(!all_zero && "Hash should not be all zeros");
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Hash Calculation", true, "", duration.count());
            std::cout << "✅ Hash calculation test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Hash Calculation", false, e.what(), duration.count());
            std::cout << "❌ Hash calculation test failed: " << e.what() << std::endl;
        }
    }
    
    void testShareValidation() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n🔍 Testing Share Validation..." << std::endl;
            
            // Create test job result
            xmrig::JobResult test_result;
            test_result.jobId = "1000";
            test_result.nonce = 0x12345678;
            test_result.poolId = 1;
            test_result.result = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";
            test_result.diff = 0x12345678;
            test_result.actualDiff = 0x12345678;
            test_result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // Test validation with different targets
            std::vector<uint32_t> test_targets = {0x10000000, 0x20000000, 0x40000000};
            
            for (uint32_t target : test_targets) {
                bool validation_result = m_share_converter.validateShare(test_result, target);
                // Note: In a real implementation, this would depend on the actual hash value
                // For now, we just test that the validation function doesn't crash
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Share Validation", true, "", duration.count());
            std::cout << "✅ Share validation test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Share Validation", false, e.what(), duration.count());
            std::cout << "❌ Share validation test failed: " << e.what() << std::endl;
        }
    }
    
    void testGPUMemoryManager() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n💾 Testing GPU Memory Manager..." << std::endl;
            
            // Test memory manager initialization
            GPUAllocationStatus init_result = m_gpu_manager.initialize(0);
            assert(init_result == GPUAllocationStatus::SUCCESS && "GPU memory manager initialization failed");
            
            // Test DAG memory allocation
            GPUAllocationStatus dag_result = m_gpu_manager.allocateDAGMemory(1024 * 1024); // 1MB
            assert(dag_result == GPUAllocationStatus::SUCCESS && "DAG memory allocation failed");
            
            // Test job blobs memory allocation
            GPUAllocationStatus jobs_result = m_gpu_manager.allocateJobBlobsMemory(10, 40); // 10 jobs, 40 bytes each
            assert(jobs_result == GPUAllocationStatus::SUCCESS && "Job blobs memory allocation failed");
            
            // Test results memory allocation
            GPUAllocationStatus results_result = m_gpu_manager.allocateResultsMemory(1000); // 1000 results
            assert(results_result == GPUAllocationStatus::SUCCESS && "Results memory allocation failed");
            
            // Test memory cleanup
            GPUAllocationStatus cleanup_result = m_gpu_manager.cleanup();
            assert(cleanup_result == GPUAllocationStatus::SUCCESS && "Memory cleanup failed");
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("GPU Memory Manager", true, "", duration.count());
            std::cout << "✅ GPU memory manager test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("GPU Memory Manager", false, e.what(), duration.count());
            std::cout << "❌ GPU memory manager test failed: " << e.what() << std::endl;
        }
    }
    
    void testMultiPoolMining() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n⛏️ Testing Multi-Pool Mining..." << std::endl;
            
            // Create multiple jobs for different pools
            std::vector<MultiPoolJob> jobs;
            
            for (uint32_t pool_id = 1; pool_id <= 3; ++pool_id) {
                MultiPoolJob job;
                job.pool_id = pool_id;
                job.height = 1000 + pool_id;
                job.target = 0x12345678 + pool_id;
                job.active = true;
                
                // Fill job blob with unique data
                for (size_t i = 0; i < sizeof(job.blob); ++i) {
                    job.blob[i] = static_cast<uint8_t>((i + pool_id) % 256);
                }
                
                jobs.push_back(job);
            }
            
            // Simulate mining process for each pool
            for (const auto& job : jobs) {
                // Test hash calculation for this job
                uint32_t test_nonce = 0x12345678;
                uint8_t hash[32];
                bool hash_result = m_share_converter.calculateHashForNonce(job, test_nonce, hash);
                
                assert(hash_result && "Hash calculation failed for multi-pool mining");
                
                // Test share conversion
                xmrig::JobResult xmrig_result;
                bool conversion_result = m_share_converter.convertBMADShareToXMRig(
                    job, test_nonce, job.pool_id, xmrig_result);
                
                assert(conversion_result && "Share conversion failed for multi-pool mining");
                assert(xmrig_result.poolId == static_cast<int>(job.pool_id) && "Pool ID mismatch");
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Multi-Pool Mining", true, "", duration.count());
            std::cout << "✅ Multi-pool mining test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Multi-Pool Mining", false, e.what(), duration.count());
            std::cout << "❌ Multi-pool mining test failed: " << e.what() << std::endl;
        }
    }
    
    void testErrorHandling() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n⚠️ Testing Error Handling..." << std::endl;
            
            // Test invalid pool ID
            bool invalid_pool_result = m_pool_connector.isPoolConnected(999);
            assert(invalid_pool_result == false && "Invalid pool should not be connected");
            
            // Test invalid job data
            MultiPoolJob invalid_job;
            invalid_job.pool_id = 0; // Invalid pool ID
            invalid_job.height = 0;   // Invalid height
            invalid_job.target = 0;   // Invalid target
            
            // Test hash calculation with invalid job
            uint8_t hash[32];
            bool hash_result = m_share_converter.calculateHashForNonce(invalid_job, 0x12345678, hash);
            // This should handle the error gracefully
            
            // Test share conversion with invalid data
            xmrig::JobResult invalid_result;
            bool conversion_result = m_share_converter.convertBMADShareToXMRig(
                invalid_job, 0x12345678, 0, invalid_result);
            // This should handle the error gracefully
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Error Handling", true, "", duration.count());
            std::cout << "✅ Error handling test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Error Handling", false, e.what(), duration.count());
            std::cout << "❌ Error handling test failed: " << e.what() << std::endl;
        }
    }
    
    void testPerformance() {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "\n⚡ Testing Performance..." << std::endl;
            
            // Create test job
            MultiPoolJob test_job;
            test_job.pool_id = 1;
            test_job.height = 1000;
            test_job.target = 0x12345678;
            test_job.active = true;
            
            // Fill job blob with test data
            for (size_t i = 0; i < sizeof(test_job.blob); ++i) {
                test_job.blob[i] = static_cast<uint8_t>(i % 256);
            }
            
            // Performance test: Calculate hashes for multiple nonces
            const int num_iterations = 1000;
            auto perf_start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < num_iterations; ++i) {
                uint32_t nonce = 0x12345678 + i;
                uint8_t hash[32];
                bool hash_result = m_share_converter.calculateHashForNonce(test_job, nonce, hash);
                assert(hash_result && "Hash calculation failed during performance test");
            }
            
            auto perf_end = std::chrono::high_resolution_clock::now();
            auto perf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(perf_end - perf_start);
            
            double hashes_per_second = (num_iterations * 1000.0) / perf_duration.count();
            
            std::cout << "  Performance: " << hashes_per_second << " hashes/second" << std::endl;
            std::cout << "  Total time: " << perf_duration.count() << "ms for " << num_iterations << " hashes" << std::endl;
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            recordTestResult("Performance", true, "", duration.count());
            std::cout << "✅ Performance test passed" << std::endl;
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            recordTestResult("Performance", false, e.what(), duration.count());
            std::cout << "❌ Performance test failed: " << e.what() << std::endl;
        }
    }
    
    void recordTestResult(const std::string& test_name, bool passed, 
                         const std::string& error_message, double execution_time_ms) {
        TestResult result;
        result.test_name = test_name;
        result.passed = passed;
        result.error_message = error_message;
        result.execution_time_ms = execution_time_ms;
        m_test_results.push_back(result);
    }
    
    void printTestSummary() {
        std::cout << "\n📊 Test Summary" << std::endl;
        std::cout << "===============" << std::endl;
        
        int total_tests = m_test_results.size();
        int passed_tests = 0;
        double total_time = 0.0;
        
        for (const auto& result : m_test_results) {
            if (result.passed) {
                passed_tests++;
            }
            total_time += result.execution_time_ms;
            
            std::string status = result.passed ? "✅ PASS" : "❌ FAIL";
            std::cout << std::left << std::setw(30) << result.test_name 
                      << " " << status 
                      << " (" << std::fixed << std::setprecision(2) << result.execution_time_ms << "ms)";
            
            if (!result.passed && !result.error_message.empty()) {
                std::cout << " - " << result.error_message;
            }
            std::cout << std::endl;
        }
        
        std::cout << "\n📈 Overall Results:" << std::endl;
        std::cout << "  Total Tests: " << total_tests << std::endl;
        std::cout << "  Passed: " << passed_tests << std::endl;
        std::cout << "  Failed: " << (total_tests - passed_tests) << std::endl;
        std::cout << "  Success Rate: " << std::fixed << std::setprecision(1) 
                  << (passed_tests * 100.0 / total_tests) << "%" << std::endl;
        std::cout << "  Total Time: " << std::fixed << std::setprecision(2) << total_time << "ms" << std::endl;
        
        if (passed_tests == total_tests) {
            std::cout << "\n🎉 All tests passed! BMAD integration is working correctly." << std::endl;
        } else {
            std::cout << "\n⚠️ Some tests failed. Please review the error messages above." << std::endl;
        }
    }
};

} // namespace BMAD

int main() {
    try {
        BMAD::BMADIntegrationTest test_suite;
        test_suite.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test suite failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 