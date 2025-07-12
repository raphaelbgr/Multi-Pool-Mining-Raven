#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256
#define HEADER_SIZE 80
#define TARGET_SIZE 32
#define POOL_DATA_SIZE (4 + 32 + 32 + 44)

// Add error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}


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

__global__ void mine_kernel(uint8_t *all_data, uint32_t *results, uint64_t start_nonce, int active_pools) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nonce = start_nonce + idx;
    uint8_t temp[80];
    uint8_t hash1[32], hash2[32];

    for(int p=0; p<active_pools; p++) {
        int pool_offset = p * POOL_DATA_SIZE;
        
        uint32_t hdr_len = *((uint32_t*)&all_data[pool_offset]);
        if(hdr_len != 32) continue;
        
        memcpy(temp, &all_data[pool_offset + 4], 32);
        
        // Insert nonce at RVN position
        temp[76] = (nonce >> 0) & 0xFF;
        temp[77] = (nonce >> 8) & 0xFF;
        temp[78] = (nonce >> 16) & 0xFF;
        temp[79] = (nonce >> 24) & 0xFF;
        
        const uint8_t *target = &all_data[pool_offset + 36];
        
        sha256(temp, 80, hash1);
        sha256(hash1, 32, hash2);
        
        bool valid = true;
        for(int i=0; i<32; i++) {
            if(hash2[i] > target[i]) {
                valid = false;
                break;
            } else if(hash2[i] < target[i]) {
                break;
            }
        }
        
        if(valid) {
            atomicCAS(&results[p], 0, (uint32_t)nonce);
        }
    }
}

typedef struct {
    uint8_t* headers;
    uint8_t* targets;
    uint32_t* results;
    int count;
} MiningData;

MiningData* create_mining_data(int pool_count) {
    MiningData* data = (MiningData*)malloc(sizeof(MiningData));
    if (!data) return NULL;

    data->count = pool_count;
    data->headers = (uint8_t*)malloc(pool_count * HEADER_SIZE);
    data->targets = (uint8_t*)malloc(pool_count * TARGET_SIZE);
    data->results = (uint32_t*)calloc(pool_count, sizeof(uint32_t));

    if (!data->headers || !data->targets || !data->results) {
        free(data->headers);
        free(data->targets);
        free(data->results);
        free(data);
        return NULL;
    }

    return data;
}

void free_mining_data(MiningData* data) {
    if (data) {
        free(data->headers);
        free(data->targets);
        free(data->results);
        free(data);
    }
}

MiningData* load_mining_data(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open data file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Calculate complete pools and remaining bytes
    const int pool_data_size = HEADER_SIZE + TARGET_SIZE;
    int complete_pools = file_size / pool_data_size;
    int remaining_bytes = file_size % pool_data_size;

    // Handle partial data by adding one more pool if there's any data left
    int total_pools = complete_pools;
    if (remaining_bytes >= HEADER_SIZE) {
        total_pools++;  // We can at least process the header
    }

    if (total_pools == 0) {
        fprintf(stderr, "Not enough data for even one complete header\n");
        fclose(file);
        return NULL;
    }

    MiningData* data = create_mining_data(total_pools);
    if (!data) {
        fclose(file);
        return NULL;
    }

    // Read complete pools
    for (int i = 0; i < complete_pools; i++) {
        if (fread(data->headers + i * HEADER_SIZE, 1, HEADER_SIZE, file) != HEADER_SIZE ||
            fread(data->targets + i * TARGET_SIZE, 1, TARGET_SIZE, file) != TARGET_SIZE) {
            fprintf(stderr, "Failed to read complete pool data\n");
            free_mining_data(data);
            fclose(file);
            return NULL;
        }
    }

    // Handle partial last pool if needed
    if (remaining_bytes > 0 && remaining_bytes >= HEADER_SIZE) {
        if (fread(data->headers + complete_pools * HEADER_SIZE, 1, HEADER_SIZE, file) != HEADER_SIZE) {
            fprintf(stderr, "Failed to read partial pool header\n");
            free_mining_data(data);
            fclose(file);
            return NULL;
        }
        // If we don't have full target data, use zeros
        memset(data->targets + complete_pools * TARGET_SIZE, 0, TARGET_SIZE);
        int target_bytes_read = fread(data->targets + complete_pools * TARGET_SIZE, 
                                    1, remaining_bytes - HEADER_SIZE, file);
        if (target_bytes_read < 0) {
            fprintf(stderr, "Error reading partial target data\n");
            free_mining_data(data);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    return data;
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
    
    // Dynamic allocation for variable-sized arrays
    uint8_t *all_data = (uint8_t*)malloc(active_pools * POOL_DATA_SIZE);
    uint32_t *results = (uint32_t*)malloc(active_pools * sizeof(uint32_t));
    
    if(!all_data || !results) {
        printf("Memory allocation failed\n");
        fclose(f);
        free(all_data);
        free(results);
        return 1;
    }
    
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
    uint32_t *d_results;
    
    CUDA_CHECK(cudaMalloc(&d_all_data, active_pools * POOL_DATA_SIZE));
    CUDA_CHECK(cudaMalloc(&d_results, active_pools * sizeof(uint32_t)));
    
    CUDA_CHECK(cudaMemcpy(d_all_data, all_data, active_pools * POOL_DATA_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_results, 0, active_pools * sizeof(uint32_t)));
    
    // Launch kernel
    int blocks = 1024;
    mine_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_all_data, d_results, start_nonce, active_pools);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get results
    CUDA_CHECK(cudaMemcpy(results, d_results, active_pools * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // Output
    bool found = false;
    for(int p=0; p<active_pools; p++) {
        if(results[p] != 0) {
            printf("Valid nonce found for pool %d: %u\n", p, results[p]);
            found = true;
        }
    }
    
    if(!found) {
        printf("No valid nonces found\n");
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_all_data));
    CUDA_CHECK(cudaFree(d_results));
    free(all_data);
    free(results);
    
    return 0;
}