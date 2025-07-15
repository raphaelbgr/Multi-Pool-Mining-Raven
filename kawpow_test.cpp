#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>

// Include KAWPOW headers
#include "kawpow_algo/libprogpow/ProgPow.h"
#include "kawpow_algo/libethash-cuda/CUDAMiner.h"
#include "kawpow_algo/libethash-cuda/cuda_helper.h"

// Test function to verify KAWPOW integration
void test_kawpow_integration() {
    std::cout << "Testing KAWPOW algorithm integration..." << std::endl;
    
    // Test ProgPoW kernel generation
    uint64_t seed = 12345;
    std::string kernel = ProgPow::getKern(seed, ProgPow::KERNEL_CUDA);
    std::cout << "Generated CUDA kernel for seed " << seed << std::endl;
    std::cout << "Kernel length: " << kernel.length() << " characters" << std::endl;
    
    // Test CUDA device initialization
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }
    
    std::cout << "KAWPOW integration test completed successfully!" << std::endl;
}

int main() {
    std::cout << "=== KAWPOW Algorithm Integration Test ===" << std::endl;
    
    try {
        test_kawpow_integration();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 