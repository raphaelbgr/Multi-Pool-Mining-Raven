#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

// X16R algorithm includes
#include "crypto/blake.cuh"
#include "crypto/bmw.cuh"
#include "crypto/groestl.cuh"
#include "crypto/jh.cuh"
#include "crypto/keccak.cuh"
#include "crypto/skein.cuh"
#include "crypto/luffa.cuh"
#include "crypto/cubehash.cuh"
#include "crypto/shavite.cuh"
#include "crypto/simd.cuh"
#include "crypto/echo.cuh"
#include "crypto/hamsi.cuh"
#include "crypto/fugue.cuh"
#include "crypto/shabal.cuh"
#include "crypto/whirlpool.cuh"
#include "crypto/sha512.cuh"
#include "crypto/haif.cuh"

#define MAX_CHALLENGES 32
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 8192

typedef struct {
    uint32_t header[20];  // 80 bytes
    uint32_t target[8];   // 32 bytes
    char pool_name[64];
    uint32_t pool_name_len;
} Challenge;

typedef struct {
    char pool_name[64];
    uint32_t nonce;
    char extranonce2[16];
} Share;

__global__ void x16r_kernel(Challenge* challenges, int num_challenges, uint32_t start_nonce, Share* shares, int* share_count) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = start_nonce + tid;
    
    // Create header with nonce
    uint32_t header[20];
    memcpy(header, challenges[0].header, sizeof(header));
    header[19] = nonce;  // Set nonce in last 4 bytes
    
    // X16R hashing sequence
    uint32_t hash[8];
    uint32_t temp_hash[8];
    
    // Initial hash (Blake)
    blake256_round(hash, header);
    
    // X16R sequence: Blake -> BMW -> Groestl -> JH -> Keccak -> Skein -> Luffa -> Cubehash -> Shavite -> Simd -> Echo -> Hamsi -> Fugue -> Shabal -> Whirlpool -> SHA512
    bmw256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    groestl256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    jh256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    keccak256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    skein256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    luffa256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    cubehash256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    shavite256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    simd256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    echo256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    hamsi256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    fugue256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    shabal256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    whirlpool256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    sha512256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    haif256_round(temp_hash, hash);
    memcpy(hash, temp_hash, sizeof(hash));
    
    // Test hash against all pool targets
    for (int i = 0; i < num_challenges; i++) {
        bool valid = true;
        
        // Compare hash with target (little-endian)
        for (int j = 7; j >= 0; j--) {
            if (temp_hash[j] > challenges[i].target[j]) {
                valid = false;
                break;
            } else if (temp_hash[j] < challenges[i].target[j]) {
                break;
            }
        }
        
        if (valid) {
            // Found valid share for this pool
            int idx = atomicAdd(share_count, 1);
            if (idx < MAX_CHALLENGES) {
                strcpy(shares[idx].pool_name, challenges[i].pool_name);
                shares[idx].nonce = nonce;
                strcpy(shares[idx].extranonce2, "00000000");  // Default extranonce2
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <start_nonce>\n", argv[0]);
        return 1;
    }
    
    uint32_t start_nonce = atoi(argv[1]);
    
    // Read challenges from file
    FILE* fp = fopen("challenges.bin", "rb");
    if (!fp) {
        printf("Error: Cannot open challenges.bin\n");
        return 1;
    }
    
    Challenge challenges[MAX_CHALLENGES];
    int num_challenges = 0;
    
    while (!feof(fp) && num_challenges < MAX_CHALLENGES) {
        // Read header length
        uint32_t header_len;
        if (fread(&header_len, sizeof(uint32_t), 1, fp) != 1) break;
        
        // Read header
        if (fread(challenges[num_challenges].header, 1, header_len, fp) != header_len) break;
        
        // Read target
        if (fread(challenges[num_challenges].target, 1, 32, fp) != 32) break;
        
        // Read pool name length
        uint32_t name_len;
        if (fread(&name_len, sizeof(uint32_t), 1, fp) != 1) break;
        
        // Read pool name
        if (fread(challenges[num_challenges].pool_name, 1, name_len, fp) != name_len) break;
        challenges[num_challenges].pool_name[name_len] = '\0';
        challenges[num_challenges].pool_name_len = name_len;
        
        num_challenges++;
    }
    
    fclose(fp);
    
    if (num_challenges == 0) {
        printf("Error: No challenges found in challenges.bin\n");
        return 1;
    }
    
    printf("Loaded %d challenges\n", num_challenges);
    for (int i = 0; i < num_challenges; i++) {
        printf("  Challenge %d: %s\n", i, challenges[i].pool_name);
    }
    
    // Allocate GPU memory
    Challenge* d_challenges;
    Share* d_shares;
    int* d_share_count;
    
    cudaMalloc(&d_challenges, sizeof(Challenge) * num_challenges);
    cudaMalloc(&d_shares, sizeof(Share) * MAX_CHALLENGES);
    cudaMalloc(&d_share_count, sizeof(int));
    
    // Copy data to GPU
    cudaMemcpy(d_challenges, challenges, sizeof(Challenge) * num_challenges, cudaMemcpyHostToDevice);
    cudaMemset(d_share_count, 0, sizeof(int));
    
    // Launch kernel
    printf("Starting X16R mining with %d challenges...\n", num_challenges);
    printf("Nonce range: %u to %u\n", start_nonce, start_nonce + THREADS_PER_BLOCK * BLOCKS_PER_GRID - 1);
    
    clock_t start = clock();
    
    x16r_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
        d_challenges, num_challenges, start_nonce, d_shares, d_share_count
    );
    
    cudaDeviceSynchronize();
    
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Get results
    int share_count;
    Share shares[MAX_CHALLENGES];
    
    cudaMemcpy(&share_count, d_share_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(shares, d_shares, sizeof(Share) * share_count, cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Mining completed in %.3f seconds\n", elapsed);
    printf("Found %d valid shares:\n", share_count);
    
    for (int i = 0; i < share_count; i++) {
        printf("SHARE %s %u %s\n", 
               shares[i].pool_name, 
               shares[i].nonce, 
               shares[i].extranonce2);
    }
    
    // Cleanup
    cudaFree(d_challenges);
    cudaFree(d_shares);
    cudaFree(d_share_count);
    
    return 0;
} 