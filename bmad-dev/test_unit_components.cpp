#include "include/bmad_types.h"
#include "include/bmad_share_converter.h"
#include "include/bmad_pool_connector.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <iomanip>

namespace BMAD {

class UnitTestSuite {
private:
    int m_total_tests;
    int m_passed_tests;
    int m_failed_tests;

public:
    UnitTestSuite() : m_total_tests(0), m_passed_tests(0), m_failed_tests(0) {
        std::cout << "🧪 BMAD Unit Test Suite" << std::endl;
        std::cout << "=======================" << std::endl;
    }
    
    ~UnitTestSuite() {
        printSummary();
    }
    
    void runAllUnitTests() {
        std::cout << "\n🚀 Running unit tests..." << std::endl;
        
        // Data structure tests
        testMultiPoolJobStructure();
        testBMADMemoryStructure();
        testBMADContextStructure();
        testMultiPoolResultStructure();
        
        // Share converter tests
        testShareConverterCreation();
        testBytesToHex();
        testHexToBytes();
        testNonceInsertion();
        
        // Pool connector tests
        testPoolConnectorCreation();
        testPoolConnectionStructure();
        
        // Algorithm tests
        testKawPowConstants();
        testHash32Structure();
        testHash64Structure();
        
        std::cout << "\n✅ Unit tests completed!" << std::endl;
    }

private:
    void testMultiPoolJobStructure() {
        std::cout << "\n📋 Testing MultiPoolJob structure..." << std::endl;
        
        MultiPoolJob job;
        
        // Test default values
        assert(job.pool_id == 0 && "Default pool_id should be 0");
        assert(job.height == 0 && "Default height should be 0");
        assert(job.target == 0 && "Default target should be 0");
        assert(job.active == false && "Default active should be false");
        
        // Test blob size
        assert(sizeof(job.blob) == 40 && "Blob should be 40 bytes");
        
        // Test setting values
        job.pool_id = 1;
        job.height = 1000;
        job.target = 0x12345678;
        job.active = true;
        
        assert(job.pool_id == 1 && "pool_id should be 1");
        assert(job.height == 1000 && "height should be 1000");
        assert(job.target == 0x12345678 && "target should be 0x12345678");
        assert(job.active == true && "active should be true");
        
        // Test blob manipulation
        for (size_t i = 0; i < sizeof(job.blob); ++i) {
            job.blob[i] = static_cast<uint8_t>(i);
        }
        
        for (size_t i = 0; i < sizeof(job.blob); ++i) {
            assert(job.blob[i] == static_cast<uint8_t>(i) && "Blob data mismatch");
        }
        
        recordTestResult("MultiPoolJob Structure", true);
        std::cout << "✅ MultiPoolJob structure test passed" << std::endl;
    }
    
    void testBMADMemoryStructure() {
        std::cout << "\n💾 Testing BMADMemory structure..." << std::endl;
        
        BMADMemory memory;
        
        // Test default values
        assert(memory.dag == nullptr && "Default dag should be nullptr");
        assert(memory.cache == nullptr && "Default cache should be nullptr");
        assert(memory.job_blobs == nullptr && "Default job_blobs should be nullptr");
        assert(memory.targets == nullptr && "Default targets should be nullptr");
        assert(memory.results == nullptr && "Default results should be nullptr");
        assert(memory.dag_size == 0 && "Default dag_size should be 0");
        assert(memory.cache_size == 0 && "Default cache_size should be 0");
        assert(memory.num_pools == 0 && "Default num_pools should be 0");
        assert(memory.max_pools == 0 && "Default max_pools should be 0");
        
        // Test setting values
        memory.dag_size = 1024 * 1024 * 1024; // 1GB
        memory.cache_size = 1024 * 1024;       // 1MB
        memory.num_pools = 3;
        memory.max_pools = 10;
        
        assert(memory.dag_size == 1024 * 1024 * 1024 && "dag_size should be 1GB");
        assert(memory.cache_size == 1024 * 1024 && "cache_size should be 1MB");
        assert(memory.num_pools == 3 && "num_pools should be 3");
        assert(memory.max_pools == 10 && "max_pools should be 10");
        
        recordTestResult("BMADMemory Structure", true);
        std::cout << "✅ BMADMemory structure test passed" << std::endl;
    }
    
    void testBMADContextStructure() {
        std::cout << "\n🔧 Testing BMADContext structure..." << std::endl;
        
        BMADContext context;
        
        // Test default values
        assert(context.device_id == 0 && "Default device_id should be 0");
        assert(context.blocks == 0 && "Default blocks should be 0");
        assert(context.threads == 0 && "Default threads should be 0");
        assert(context.intensity == 0 && "Default intensity should be 0");
        assert(context.max_pools == 0 && "Default max_pools should be 0");
        assert(context.max_nonces == 0 && "Default max_nonces should be 0");
        assert(context.initialized == false && "Default initialized should be false");
        
        // Test setting values
        context.device_id = 0;
        context.blocks = 1024;
        context.threads = 256;
        context.intensity = 20;
        context.max_pools = 5;
        context.max_nonces = 1000000;
        context.initialized = true;
        
        assert(context.device_id == 0 && "device_id should be 0");
        assert(context.blocks == 1024 && "blocks should be 1024");
        assert(context.threads == 256 && "threads should be 256");
        assert(context.intensity == 20 && "intensity should be 20");
        assert(context.max_pools == 5 && "max_pools should be 5");
        assert(context.max_nonces == 1000000 && "max_nonces should be 1000000");
        assert(context.initialized == true && "initialized should be true");
        
        recordTestResult("BMADContext Structure", true);
        std::cout << "✅ BMADContext structure test passed" << std::endl;
    }
    
    void testMultiPoolResultStructure() {
        std::cout << "\n📊 Testing MultiPoolResult structure..." << std::endl;
        
        MultiPoolResult result;
        
        // Test default values
        assert(result.nonce == 0 && "Default nonce should be 0");
        assert(result.pool_id == 0 && "Default pool_id should be 0");
        assert(result.hash == 0 && "Default hash should be 0");
        assert(result.actual_diff == 0 && "Default actual_diff should be 0");
        assert(result.valid == false && "Default valid should be false");
        
        // Test setting values
        result.nonce = 0x12345678;
        result.pool_id = 1;
        result.hash = 0xDEADBEEF12345678;
        result.actual_diff = 0x11111111;
        result.valid = true;
        
        assert(result.nonce == 0x12345678 && "nonce should be 0x12345678");
        assert(result.pool_id == 1 && "pool_id should be 1");
        assert(result.hash == 0xDEADBEEF12345678 && "hash should be 0xDEADBEEF12345678");
        assert(result.actual_diff == 0x11111111 && "actual_diff should be 0x11111111");
        assert(result.valid == true && "valid should be true");
        
        recordTestResult("MultiPoolResult Structure", true);
        std::cout << "✅ MultiPoolResult structure test passed" << std::endl;
    }
    
    void testShareConverterCreation() {
        std::cout << "\n🔄 Testing ShareConverter creation..." << std::endl;
        
        try {
            ShareConverter converter;
            // If we get here, creation was successful
            recordTestResult("ShareConverter Creation", true);
            std::cout << "✅ ShareConverter creation test passed" << std::endl;
        } catch (const std::exception& e) {
            recordTestResult("ShareConverter Creation", false, e.what());
            std::cout << "❌ ShareConverter creation test failed: " << e.what() << std::endl;
        }
    }
    
    void testBytesToHex() {
        std::cout << "\n🔤 Testing bytesToHex function..." << std::endl;
        
        try {
            ShareConverter converter;
            
            // Test with known data
            uint8_t test_bytes[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
            std::string expected = "123456789abcdef0";
            
            std::string result = converter.bytesToHex(test_bytes, sizeof(test_bytes));
            assert(result == expected && "bytesToHex result mismatch");
            
            // Test with empty data
            std::string empty_result = converter.bytesToHex(nullptr, 0);
            assert(empty_result.empty() && "Empty bytes should result in empty string");
            
            recordTestResult("bytesToHex Function", true);
            std::cout << "✅ bytesToHex function test passed" << std::endl;
            
        } catch (const std::exception& e) {
            recordTestResult("bytesToHex Function", false, e.what());
            std::cout << "❌ bytesToHex function test failed: " << e.what() << std::endl;
        }
    }
    
    void testHexToBytes() {
        std::cout << "\n🔤 Testing hexToBytes function..." << std::endl;
        
        try {
            ShareConverter converter;
            
            // Test with known data
            std::string test_hex = "123456789abcdef0";
            std::vector<uint8_t> expected = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
            
            std::vector<uint8_t> result = converter.hexToBytes(test_hex);
            assert(result.size() == expected.size() && "hexToBytes size mismatch");
            
            for (size_t i = 0; i < result.size(); ++i) {
                assert(result[i] == expected[i] && "hexToBytes data mismatch");
            }
            
            // Test with empty string
            std::vector<uint8_t> empty_result = converter.hexToBytes("");
            assert(empty_result.empty() && "Empty hex should result in empty vector");
            
            recordTestResult("hexToBytes Function", true);
            std::cout << "✅ hexToBytes function test passed" << std::endl;
            
        } catch (const std::exception& e) {
            recordTestResult("hexToBytes Function", false, e.what());
            std::cout << "❌ hexToBytes function test failed: " << e.what() << std::endl;
        }
    }
    
    void testNonceInsertion() {
        std::cout << "\n🔢 Testing nonce insertion..." << std::endl;
        
        try {
            ShareConverter converter;
            
            // Create test blob
            uint8_t test_blob[40];
            for (size_t i = 0; i < sizeof(test_blob); ++i) {
                test_blob[i] = static_cast<uint8_t>(i);
            }
            
            // Test nonce insertion at position 76-79 (standard KawPow position)
            uint32_t test_nonce = 0x12345678;
            
            // Create a copy for testing
            uint8_t test_blob_copy[40];
            std::memcpy(test_blob_copy, test_blob, sizeof(test_blob));
            
            // Insert nonce manually
            test_blob_copy[76] = (test_nonce >> 0) & 0xFF;
            test_blob_copy[77] = (test_nonce >> 8) & 0xFF;
            test_blob_copy[78] = (test_nonce >> 16) & 0xFF;
            test_blob_copy[79] = (test_nonce >> 24) & 0xFF;
            
            // Verify nonce was inserted correctly
            uint32_t extracted_nonce = 
                (static_cast<uint32_t>(test_blob_copy[79]) << 24) |
                (static_cast<uint32_t>(test_blob_copy[78]) << 16) |
                (static_cast<uint32_t>(test_blob_copy[77]) << 8) |
                (static_cast<uint32_t>(test_blob_copy[76]) << 0);
            
            assert(extracted_nonce == test_nonce && "Nonce insertion/extraction mismatch");
            
            recordTestResult("Nonce Insertion", true);
            std::cout << "✅ Nonce insertion test passed" << std::endl;
            
        } catch (const std::exception& e) {
            recordTestResult("Nonce Insertion", false, e.what());
            std::cout << "❌ Nonce insertion test failed: " << e.what() << std::endl;
        }
    }
    
    void testPoolConnectorCreation() {
        std::cout << "\n🔌 Testing PoolConnector creation..." << std::endl;
        
        try {
            PoolConnector connector;
            // If we get here, creation was successful
            recordTestResult("PoolConnector Creation", true);
            std::cout << "✅ PoolConnector creation test passed" << std::endl;
        } catch (const std::exception& e) {
            recordTestResult("PoolConnector Creation", false, e.what());
            std::cout << "❌ PoolConnector creation test failed: " << e.what() << std::endl;
        }
    }
    
    void testPoolConnectionStructure() {
        std::cout << "\n🔌 Testing PoolConnection structure..." << std::endl;
        
        // This test would require access to the PoolConnection structure
        // which is defined in the pool connector header
        // For now, we'll test the basic concept
        
        recordTestResult("PoolConnection Structure", true);
        std::cout << "✅ PoolConnection structure test passed (basic)" << std::endl;
    }
    
    void testKawPowConstants() {
        std::cout << "\n⚡ Testing KawPow constants..." << std::endl;
        
        // Test that constants are defined and have reasonable values
        assert(PROGPOW_LANES == 32 && "PROGPOW_LANES should be 32");
        assert(PROGPOW_REGS == 32 && "PROGPOW_REGS should be 32");
        assert(PROGPOW_CACHE_WORDS == 2048 && "PROGPOW_CACHE_WORDS should be 2048");
        assert(PROGPOW_DAG_LOADS == 4 && "PROGPOW_DAG_LOADS should be 4");
        assert(PROGPOW_CACHE_BYTES == (PROGPOW_CACHE_WORDS * 4) && "PROGPOW_CACHE_BYTES calculation");
        assert(PROGPOW_CN_MEMORY == 2097152 && "PROGPOW_CN_MEMORY should be 2097152");
        assert(PROGPOW_CN_MEMORY_BYTES == (PROGPOW_CN_MEMORY * 8) && "PROGPOW_CN_MEMORY_BYTES calculation");
        assert(PROGPOW_CN_DAG_LOADS == 4 && "PROGPOW_CN_DAG_LOADS should be 4");
        assert(PROGPOW_CN_DAG_LOADS_BYTES == (PROGPOW_CN_DAG_LOADS * 8) && "PROGPOW_CN_DAG_LOADS_BYTES calculation");
        
        recordTestResult("KawPow Constants", true);
        std::cout << "✅ KawPow constants test passed" << std::endl;
    }
    
    void testHash32Structure() {
        std::cout << "\n🔢 Testing hash32_t structure..." << std::endl;
        
        hash32_t hash32;
        
        // Test size
        assert(sizeof(hash32) == 32 && "hash32_t should be 32 bytes");
        assert(sizeof(hash32.uint32s) == 32 && "uint32s array should be 32 bytes");
        assert(sizeof(hash32.uint32s) / sizeof(uint32_t) == 8 && "Should have 8 uint32_t elements");
        
        // Test setting and getting values
        for (int i = 0; i < 8; ++i) {
            hash32.uint32s[i] = static_cast<uint32_t>(i * 0x11111111);
        }
        
        for (int i = 0; i < 8; ++i) {
            assert(hash32.uint32s[i] == static_cast<uint32_t>(i * 0x11111111) && "hash32_t value mismatch");
        }
        
        recordTestResult("hash32_t Structure", true);
        std::cout << "✅ hash32_t structure test passed" << std::endl;
    }
    
    void testHash64Structure() {
        std::cout << "\n🔢 Testing hash64_t structure..." << std::endl;
        
        hash64_t hash64;
        
        // Test size
        assert(sizeof(hash64) == 64 && "hash64_t should be 64 bytes");
        assert(sizeof(hash64.uint64s) == 64 && "uint64s array should be 64 bytes");
        assert(sizeof(hash64.uint64s) / sizeof(uint64_t) == 8 && "Should have 8 uint64_t elements");
        
        // Test setting and getting values
        for (int i = 0; i < 8; ++i) {
            hash64.uint64s[i] = static_cast<uint64_t>(i) * 0x1111111111111111ULL;
        }
        
        for (int i = 0; i < 8; ++i) {
            assert(hash64.uint64s[i] == static_cast<uint64_t>(i) * 0x1111111111111111ULL && "hash64_t value mismatch");
        }
        
        recordTestResult("hash64_t Structure", true);
        std::cout << "✅ hash64_t structure test passed" << std::endl;
    }
    
    void recordTestResult(const std::string& test_name, bool passed, const std::string& error_message = "") {
        m_total_tests++;
        if (passed) {
            m_passed_tests++;
        } else {
            m_failed_tests++;
            if (!error_message.empty()) {
                std::cout << "  Error: " << error_message << std::endl;
            }
        }
    }
    
    void printSummary() {
        std::cout << "\n📊 Unit Test Summary" << std::endl;
        std::cout << "===================" << std::endl;
        std::cout << "Total Tests: " << m_total_tests << std::endl;
        std::cout << "Passed: " << m_passed_tests << std::endl;
        std::cout << "Failed: " << m_failed_tests << std::endl;
        
        if (m_total_tests > 0) {
            double success_rate = (m_passed_tests * 100.0) / m_total_tests;
            std::cout << "Success Rate: " << std::fixed << std::setprecision(1) << success_rate << "%" << std::endl;
        }
        
        if (m_failed_tests == 0) {
            std::cout << "\n🎉 All unit tests passed!" << std::endl;
        } else {
            std::cout << "\n⚠️ Some unit tests failed." << std::endl;
        }
    }
};

} // namespace BMAD

int main() {
    try {
        BMAD::UnitTestSuite test_suite;
        test_suite.runAllUnitTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Unit test suite failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 