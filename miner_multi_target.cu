#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>

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

__device__ __constant__ uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
    0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
    0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
    0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
    0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
    0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ void sha256_transform(const uint8_t *message, uint32_t *digest) {
    uint32_t w[64];
    uint32_t a,b,c,d,e,f,g,h;
    
    #pragma unroll
    for(int i=0; i<16; i++) {
        w[i] = (message[i*4]<<24)|(message[i*4+1]<<16)|(message[i*4+2]<<8)|(message[i*4+3]);
    }
    
    #pragma unroll
    for(int i=16; i<64; i++) {
        uint32_t s0 = __funnelshift_r(w[i-15],w[i-15],7)^__funnelshift_r(w[i-15],w[i-15],18)^(w[i-15]>>3);
        uint32_t s1 = __funnelshift_r(w[i-2],w[i-2],17)^__funnelshift_r(w[i-2],w[i-2],19)^(w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    a=digest[0]; b=digest[1]; c=digest[2]; d=digest[3];
    e=digest[4]; f=digest[5]; g=digest[6]; h=digest[7];
    
    #pragma unroll
    for(int i=0; i<64; i++) {
        uint32_t S1 = __funnelshift_r(e,e,6)^__funnelshift_r(e,e,11)^__funnelshift_r(e,e,25);
        uint32_t ch = (e&f)^(~e&g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0 = __funnelshift_r(a,a,2)^__funnelshift_r(a,a,13)^__funnelshift_r(a,a,22);
        uint32_t maj = (a&b)^(a&c)^(b&c);
        uint32_t temp2 = S0 + maj;
        
        h=g; g=f; f=e; e=d+temp1;
        d=c; c=b; b=a; a=temp1+temp2;
    }
    
    digest[0]+=a; digest[1]+=b; digest[2]+=c; digest[3]+=d;
    digest[4]+=e; digest[5]+=f; digest[6]+=g; digest[7]+=h;
}

__device__ void sha256(const uint8_t *input, size_t len, uint8_t *output) {
    uint8_t block[64] = {0};
    memcpy(block, input, len);
    block[len] = 0x80;
    
    uint32_t digest[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    
    sha256_transform(block, digest);
    
    #pragma unroll
    for(int i=0; i<8; i++) {
        output[i*4+0] = (digest[i]>>24)&0xFF;
        output[i*4+1] = (digest[i]>>16)&0xFF;
        output[i*4+2] = (digest[i]>>8)&0xFF;
        output[i*4+3] = digest[i]&0xFF;
    }
}

// OPTIMIZED KERNEL: Test each nonce against ALL pool targets!
__global__ void multi_pool_mine_kernel(uint8_t *all_data, MultiPoolResult *results, 
                                       uint64_t start_nonce, int active_pools, int max_results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nonce = start_nonce + idx;
    uint8_t temp[80];
    uint8_t hash1[32], hash2[32];
    
    // We'll compute the hash once and test against ALL pools
    uint32_t valid_pools_mask = 0;
    
    // Use the first pool's header as the base (they should all have same block template)
    int pool_offset = 0;
    uint32_t hdr_len = *((uint32_t*)&all_data[pool_offset]);
    if(hdr_len != 80) return;  // Skip if invalid header
    
    memcpy(temp, &all_data[pool_offset + 4], 80);  // Copy full 80-byte header
    
    // Insert nonce at RVN position (bytes 76-79, little endian)
    temp[76] = (nonce >> 0) & 0xFF;
    temp[77] = (nonce >> 8) & 0xFF;
    temp[78] = (nonce >> 16) & 0xFF;
    temp[79] = (nonce >> 24) & 0xFF;
    
    // Calculate hash ONCE
    sha256(temp, 80, hash1);
    sha256(hash1, 32, hash2);
    
    // Now test this hash against ALL pool targets!
    for(int p = 0; p < active_pools && p < MAX_POOLS; p++) {
        int target_offset = p * POOL_DATA_SIZE + 4 + 80;  // After header_len + header
        const uint8_t *target = &all_data[target_offset];
        
        // Check if hash meets this pool's target
        bool valid = true;
        for(int i = 0; i < 32; i++) {
            if(hash2[i] > target[i]) {
                valid = false;
                break;
            } else if(hash2[i] < target[i]) {
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
    
    printf("[MULTI-POOL] Testing each nonce against %d pools simultaneously!\n", active_pools);
    
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
    
    // Initialize results
    memset(results, 0, max_results * sizeof(MultiPoolResult));
    
    size_t read = fread(all_data, 1, active_pools * POOL_DATA_SIZE, f);
    fclose(f);
    
    if(read != (size_t)(active_pools * POOL_DATA_SIZE)) {
        printf("Error reading pool data\n");
        free(all_data);
        free(results);
        return 1;
    }
    
    // GPU memory
    uint8_t *d_all_data;
    MultiPoolResult *d_results;
    
    CUDA_CHECK(cudaMalloc(&d_all_data, active_pools * POOL_DATA_SIZE));
    CUDA_CHECK(cudaMalloc(&d_results, max_results * sizeof(MultiPoolResult)));
    
    CUDA_CHECK(cudaMemcpy(d_all_data, all_data, active_pools * POOL_DATA_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_results, 0, max_results * sizeof(MultiPoolResult)));
    
    // Launch kernel with more threads for better coverage
    int blocks = 2048;  // More blocks for better efficiency
    printf("[LAUNCH] %d blocks x %d threads = %d total threads\n", 
           blocks, THREADS_PER_BLOCK, blocks * THREADS_PER_BLOCK);
    
    multi_pool_mine_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_all_data, d_results, start_nonce, active_pools, max_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get results
    CUDA_CHECK(cudaMemcpy(results, d_results, max_results * sizeof(MultiPoolResult), cudaMemcpyDeviceToHost));
    
    // Output results
    bool found = false;
    printf("\n[MULTI-POOL RESULTS]\n");
    printf("==================================================\n");
    
    for(int i = 0; i < max_results; i++) {
        if(results[i].nonce != 0) {
            found = true;
            printf("\n[JACKPOT] Nonce %u is valid for pools: ", results[i].nonce);
            
            int valid_count = 0;
            for(int p = 0; p < active_pools; p++) {
                if(results[i].valid_pools & (1 << p)) {
                    if(valid_count > 0) printf(", ");
                    printf("%d", p);
                    valid_count++;
                }
            }
            printf(" (%d pools total!)", valid_count);
            
            // Output in format expected by submit script
            for(int p = 0; p < active_pools; p++) {
                if(results[i].valid_pools & (1 << p)) {
                    printf("\nValid nonce found for pool %d: %u", p, results[i].nonce);
                }
            }
        }
    }
    
    if(!found) {
        printf("\nNo valid nonces found in this range\n");
        printf("[TIP] Try increasing the nonce range or checking pool difficulties\n");
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_all_data));
    CUDA_CHECK(cudaFree(d_results));
    free(all_data);
    free(results);
    
    return 0;
} 