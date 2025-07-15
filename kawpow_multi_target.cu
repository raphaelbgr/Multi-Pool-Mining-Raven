#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREADS_PER_BLOCK 256
#define HEADER_SIZE 80
#define TARGET_SIZE 32
#define POOL_DATA_SIZE (4 + 80 + 32)  // header_len(4) + header(80) + target(32)
#define MAX_POOLS 32  // Support up to 32 pools

// Add error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Structure to hold multi-pool results
typedef struct {
    uint32_t nonce;
    uint32_t valid_pools;  // Bitmask: which pools this nonce is valid for
} MultiPoolResult;

// Simple FNV hash function (used in ProgPoW/KAWPOW)
__device__ __host__ uint32_t fnv1a(uint32_t h, uint32_t d) {
    return (h * 0x01000193) ^ d;
}

// Simple Keccak-256 implementation (simplified for now)
__device__ __host__ void keccak256(uint32_t* output, const uint32_t* input, size_t len) {
    // Simplified Keccak - this is just a placeholder
    // In real implementation, you'd use the full Keccak-256 algorithm
    for (int i = 0; i < 8; i++) {
        output[i] = input[i] ^ 0x5a5a5a5a; // Simple XOR for demo
    }
}

// Simple ProgPoW kernel (simplified version)
__device__ void kawpow_hash(uint32_t* header, uint32_t nonce, uint32_t* output) {
    // Simplified KAWPOW algorithm
    uint32_t state[8];
    
    // Initialize state with header + nonce
    for (int i = 0; i < 8; i++) {
        state[i] = header[i];
    }
    state[0] ^= nonce; // Add nonce to first word
    
    // Simple mixing rounds (simplified)
    for (int round = 0; round < 64; round++) {
        uint32_t mix = fnv1a(state[round % 8], round);
        state[round % 8] = fnv1a(state[round % 8], mix);
    }
    
    // Final hash
    keccak256(output, state, 32);
}

// KAWPOW Multi-Pool Mining Kernel
__global__ void kawpow_multi_pool_mine_kernel(uint8_t *all_data, MultiPoolResult *results, 
                                              uint64_t start_nonce, int active_pools, int max_results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nonce = start_nonce + idx;
    uint32_t header[8];
    uint32_t hash_result[8];
    
    // We'll compute the KAWPOW hash once and test against ALL pools
    uint32_t valid_pools_mask = 0;
    
    // Use the first pool's header as the base (they should all have same block template)
    int pool_offset = 0;
    uint32_t hdr_len = *((uint32_t*)&all_data[pool_offset]);
    if(hdr_len != 80) return;  // Skip if invalid header
    
    // Convert 80-byte header to 8 uint32_t words for KAWPOW
    for (int i = 0; i < 8; i++) {
        header[i] = ((uint32_t)all_data[pool_offset + 4 + i*4 + 0] << 24) |
                   ((uint32_t)all_data[pool_offset + 4 + i*4 + 1] << 16) |
                   ((uint32_t)all_data[pool_offset + 4 + i*4 + 2] << 8) |
                   ((uint32_t)all_data[pool_offset + 4 + i*4 + 3]);
    }
    
    // Calculate KAWPOW hash
    kawpow_hash(header, (uint32_t)nonce, hash_result);
    
    // Now test this hash against ALL pool targets!
    for(int p = 0; p < active_pools && p < MAX_POOLS; p++) {
        int target_offset = p * POOL_DATA_SIZE + 4 + 80;  // After header_len + header
        const uint8_t *target = &all_data[target_offset];
        
        // Convert target to uint32_t for comparison
        uint32_t target_words[8];
        for (int i = 0; i < 8; i++) {
            target_words[i] = ((uint32_t)target[i*4 + 0] << 24) |
                             ((uint32_t)target[i*4 + 1] << 16) |
                             ((uint32_t)target[i*4 + 2] << 8) |
                             ((uint32_t)target[i*4 + 3]);
        }
        
        // Check if hash meets this pool's target (reverse byte order for comparison)
        bool valid = true;
        for(int i = 7; i >= 0; i--) {  // Start from most significant word
            if(hash_result[i] > target_words[i]) {
                valid = false;
                break;
            } else if(hash_result[i] < target_words[i]) {
                break;  // Hash is definitely lower, it's valid
            }
        }
        
        if(valid) {
            valid_pools_mask |= (1 << p);  // Set bit for this pool
        }
    }
    
    // If we found a valid nonce for any pool, try to store it
    if(valid_pools_mask != 0) {
        // Find an empty slot in results array
        for(int i = 0; i < max_results; i++) {
            if(atomicCAS(&results[i].nonce, 0, (uint32_t)nonce) == 0) {
                results[i].valid_pools = valid_pools_mask;
                break;
            }
        }
    }
}

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: %s <start_nonce>\n", argv[0]);
        return 1;
    }
    
    uint64_t start_nonce = strtoull(argv[1], NULL, 10);
    
    // Load data
    FILE *f = fopen("headers.bin", "rb");
    if(!f) {
        printf("Error opening headers.bin\n");
        return 1;
    }
    
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    int active_pools = file_size / POOL_DATA_SIZE;
    if(active_pools == 0) {
        printf("No complete pools found\n");
        fclose(f);
        return 1;
    }
    
    printf("[KAWPOW MULTI-POOL] Testing each nonce against %d pools simultaneously!\n", active_pools);
    
    // Dynamic allocation for variable-sized arrays
    uint8_t *all_data = (uint8_t*)malloc(active_pools * POOL_DATA_SIZE);
    const int max_results = 1000;  // Store up to 1000 valid nonces
    MultiPoolResult *results = (MultiPoolResult*)malloc(max_results * sizeof(MultiPoolResult));
    
    if(!all_data || !results) {
        printf("Memory allocation failed\n");
        fclose(f);
        free(all_data);
        free(results);
        return 1;
    }
    
    // Initialize results array
    memset(results, 0, max_results * sizeof(MultiPoolResult));
    
    // Read all pool data
    size_t bytes_read = fread(all_data, 1, active_pools * POOL_DATA_SIZE, f);
    fclose(f);
    
    if(bytes_read != active_pools * POOL_DATA_SIZE) {
        printf("Error reading pool data\n");
        free(all_data);
        free(results);
        return 1;
    }
    
    // Allocate GPU memory
    uint8_t *d_all_data;
    MultiPoolResult *d_results;
    
    CUDA_CHECK(cudaMalloc(&d_all_data, active_pools * POOL_DATA_SIZE));
    CUDA_CHECK(cudaMalloc(&d_results, max_results * sizeof(MultiPoolResult)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_all_data, all_data, active_pools * POOL_DATA_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_results, results, max_results * sizeof(MultiPoolResult), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (1000000 + threadsPerBlock - 1) / threadsPerBlock;  // Test 1M nonces
    
    printf("Launching KAWPOW kernel with %d blocks, %d threads per block\n", blocksPerGrid, threadsPerBlock);
    printf("Testing nonces from %llu to %llu\n", start_nonce, start_nonce + 1000000 - 1);
    
    kawpow_multi_pool_mine_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_all_data, d_results, start_nonce, active_pools, max_results
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(results, d_results, max_results * sizeof(MultiPoolResult), cudaMemcpyDeviceToHost));
    
    // Process results
    int valid_count = 0;
    for(int i = 0; i < max_results; i++) {
        if(results[i].nonce != 0) {
            valid_count++;
            printf("JACKPOT! Nonce %u valid for pools: ", results[i].nonce);
            
            // Print which pools this nonce is valid for
            for(int p = 0; p < active_pools; p++) {
                if(results[i].valid_pools & (1 << p)) {
                    printf("%d ", p);
                }
            }
            printf("\n");
        }
    }
    
    if(valid_count == 0) {
        printf("No valid nonces found in this range\n");
    } else {
        printf("Found %d valid nonces!\n", valid_count);
    }
    
    // Cleanup
    cudaFree(d_all_data);
    cudaFree(d_results);
    free(all_data);
    free(results);
    
    return 0;
} 