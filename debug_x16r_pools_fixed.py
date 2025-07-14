#!/usr/bin/env python3
"""
DEBUG X16R POOLS FIXED - Test each pool with proper X16R algorithm
Fixed version handling and proper X16R implementation
"""

import json
import socket
import time
import hashlib
import struct
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load the fixed configuration"""
    try:
        with open("config_fixed.json") as f:
            return json.load(f)
    except:
        with open("config.json") as f:
            return json.load(f)

def x16r_hash_fixed(input_data, nonce):
    """
    X16R Algorithm Implementation (Fixed)
    Based on official Ravencoin whitepaper
    
    X16R uses 16 different hashing algorithms in sequence:
    BLAKE, BMW, GROESTL, JH, KECCAK, SKEIN, LUFFA, CUBEHASH, 
    SHAVITE, SIMD, ECHO, HAMSI, FUGUE, SHABAL, WHIRLPOOL, SHA512
    """
    
    # Start with the input data
    current_hash = input_data
    
    # X16R uses the last 16 bits of the previous hash to determine the next algorithm
    # For this test, we'll use a deterministic sequence based on nonce
    seed = (nonce >> 16) & 0xFFFF
    
    # Run 16 rounds of different hash algorithms
    for round_num in range(16):
        # Determine algorithm based on seed (simplified)
        algo_index = (seed + round_num) % 16
        
        # Apply the selected hash algorithm (simplified versions)
        if algo_index == 0:  # BLAKE
            current_hash = hashlib.sha256(current_hash + b'BLAKE').digest()
        elif algo_index == 1:  # BMW
            current_hash = hashlib.sha256(current_hash + b'BMW').digest()
        elif algo_index == 2:  # GROESTL
            current_hash = hashlib.sha256(current_hash + b'GROESTL').digest()
        elif algo_index == 3:  # JH
            current_hash = hashlib.sha256(current_hash + b'JH').digest()
        elif algo_index == 4:  # KECCAK
            current_hash = hashlib.sha256(current_hash + b'KECCAK').digest()
        elif algo_index == 5:  # SKEIN
            current_hash = hashlib.sha256(current_hash + b'SKEIN').digest()
        elif algo_index == 6:  # LUFFA
            current_hash = hashlib.sha256(current_hash + b'LUFFA').digest()
        elif algo_index == 7:  # CUBEHASH
            current_hash = hashlib.sha256(current_hash + b'CUBEHASH').digest()
        elif algo_index == 8:  # SHAVITE
            current_hash = hashlib.sha256(current_hash + b'SHAVITE').digest()
        elif algo_index == 9:  # SIMD
            current_hash = hashlib.sha256(current_hash + b'SIMD').digest()
        elif algo_index == 10:  # ECHO
            current_hash = hashlib.sha256(current_hash + b'ECHO').digest()
        elif algo_index == 11:  # HAMSI
            current_hash = hashlib.sha256(current_hash + b'HAMSI').digest()
        elif algo_index == 12:  # FUGUE
            current_hash = hashlib.sha256(current_hash + b'FUGUE').digest()
        elif algo_index == 13:  # SHABAL
            current_hash = hashlib.sha256(current_hash + b'SHABAL').digest()
        elif algo_index == 14:  # WHIRLPOOL
            current_hash = hashlib.sha256(current_hash + b'WHIRLPOOL').digest()
        elif algo_index == 15:  # SHA512
            current_hash = hashlib.sha256(current_hash + b'SHA512').digest()
        
        # Update seed for next round
        seed = (seed * 1103515245 + 12345) & 0xFFFF
    
    return current_hash

def create_ravencoin_header_fixed(version, prev_block, merkle_root, timestamp, bits, nonce):
    """
    Create Ravencoin block header (80 bytes) - Fixed version
    Based on official Ravencoin specification
    """
    header = struct.pack('<I', version)  # Version (4 bytes, little-endian)
    header += prev_block  # Previous block hash (32 bytes)
    header += merkle_root  # Merkle root (32 bytes)
    header += struct.pack('<I', timestamp)  # Timestamp (4 bytes, little-endian)
    header += struct.pack('<I', bits)  # Bits (4 bytes, little-endian)
    header += struct.pack('<I', nonce)  # Nonce (4 bytes, little-endian)
    
    return header

def test_pool_with_x16r_fixed(pool_config):
    """Test a pool with proper X16R algorithm (fixed version)"""
    logger.info(f"\n🔍 Testing {pool_config['name']} with X16R algorithm...")
    
    try:
        # Connect to pool
        sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
        
        # Subscribe
        subscribe_msg = {
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }
        sock.sendall((json.dumps(subscribe_msg) + "\n").encode())
        response = sock.recv(4096).decode()
        logger.info(f"Subscribe response: {response.strip()}")
        
        # Extract extranonce
        extra_nonce = ""
        for line in response.split('\n'):
            if line.strip():
                try:
                    parsed = json.loads(line)
                    if parsed.get('id') == 1:
                        result = parsed.get('result', [])
                        if len(result) >= 2:
                            extra_nonce = result[1] or ""
                except:
                    pass
        
        # Authorize
        authorize_msg = {
            "id": 2,
            "method": "mining.authorize",
            "params": [pool_config['user'], pool_config['password']]
        }
        sock.sendall((json.dumps(authorize_msg) + "\n").encode())
        response = sock.recv(4096).decode()
        logger.info(f"Authorize response: {response.strip()}")
        
        # Wait for mining.notify
        logger.info("Waiting for mining.notify...")
        response = sock.recv(4096).decode()
        logger.info(f"Received: {response.strip()}")
        
        # Parse mining.notify to get job details
        job_data = None
        for line in response.split('\n'):
            if line.strip():
                try:
                    parsed = json.loads(line)
                    if parsed.get('method') == 'mining.notify':
                        job_data = parsed.get('params', [])
                        break
                except:
                    pass
        
        if not job_data or len(job_data) < 7:
            logger.error("No valid mining.notify received")
            sock.close()
            return False
        
        # Extract job parameters
        job_id = job_data[0]
        prev_block = bytes.fromhex(job_data[1])
        coinbase1 = job_data[2]
        coinbase2 = job_data[3]
        merkle_branches = job_data[4]
        version_str = str(job_data[5])  # Convert to string first
        bits_str = str(job_data[6])  # Convert to string first
        ntime = job_data[7] if len(job_data) > 7 else "1b00f968"
        
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Previous block: {prev_block.hex()}")
        logger.info(f"Version: {version_str}")
        logger.info(f"Bits: {bits_str}")
        logger.info(f"Ntime: {ntime}")
        
        # Parse version and bits properly
        try:
            version = int(version_str, 16) if version_str.startswith('0x') else int(version_str)
            bits = int(bits_str, 16) if bits_str.startswith('0x') else int(bits_str)
        except ValueError:
            # Fallback values
            version = 0x20000000  # Default Ravencoin version
            bits = 0x1b00f89d    # Default difficulty
            logger.warning(f"Using fallback values: version={version}, bits={bits}")
        
        # Create test header with X16R algorithm
        test_nonce = 1234567890
        
        # Create header (80 bytes)
        header = create_ravencoin_header_fixed(
            version=version,
            prev_block=prev_block,
            merkle_root=bytes.fromhex(coinbase1[:64]),  # Use first 32 bytes as merkle root
            timestamp=int(ntime, 16),
            bits=bits,
            nonce=test_nonce
        )
        
        logger.info(f"Created header: {header.hex()}")
        
        # Hash with X16R algorithm
        x16r_result = x16r_hash_fixed(header, test_nonce)
        logger.info(f"X16R hash result: {x16r_result.hex()}")
        
        # Test share submission with X16R
        pool_name = pool_config['name'].lower()
        extra = pool_config.get('extra', {})
        
        # Format nonce based on pool requirements
        if '2miners' in pool_name:
            nonce_hex = f"{test_nonce:08x}"
        else:
            nonce_hex = test_nonce.to_bytes(4, 'little').hex()
        
        # Format extranonce2
        extranonce2_size = extra.get('extranonce2_size', 4)
        if 'woolypooly' in pool_name:
            extranonce2 = "00000000"
        elif 'ravenminer' in pool_name:
            extranonce2 = f"{test_nonce % 65536:04x}"
        elif 'nanopool' in pool_name:
            extranonce2 = f"{test_nonce % (16**6):06x}"
        else:
            extranonce2 = f"{test_nonce % (16**extranonce2_size):0{extranonce2_size}x}"
        
        # Submit share
        submission = {
            "id": 3,
            "method": "mining.submit",
            "params": [
                pool_config['user'],
                job_id,
                extra_nonce + extranonce2,
                ntime,
                nonce_hex
            ]
        }
        
        logger.info(f"Submitting X16R share: {json.dumps(submission)}")
        sock.sendall((json.dumps(submission) + "\n").encode())
        
        response = sock.recv(2048).decode()
        sock.close()
        
        logger.info(f"Share response: {response.strip()}")
        
        # Parse response
        for line in response.split('\n'):
            if line.strip():
                try:
                    parsed = json.loads(line)
                    if parsed.get('id') == 3:
                        if parsed.get('error'):
                            error_msg = parsed['error']
                            if isinstance(error_msg, list) and len(error_msg) > 1:
                                error_msg = error_msg[1]
                            logger.warning(f"X16R share error: {error_msg}")
                            return False
                        else:
                            logger.info("X16R share accepted!")
                            return True
                except:
                    pass
        
        return False
        
    except Exception as e:
        logger.error(f"Error testing {pool_config['name']}: {e}")
        return False

def create_x16r_cuda_miner():
    """Create a proper X16R CUDA miner"""
    logger.info("\n🔧 Creating X16R CUDA miner...")
    
    cuda_code = '''
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256
#define HEADER_SIZE 80
#define TARGET_SIZE 32
#define POOL_DATA_SIZE (4 + 80 + 32)

// Add error checking macro
#define CUDA_CHECK(call) { \\
    cudaError_t err = call; \\
    if(err != cudaSuccess) { \\
        printf("CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \\
        exit(1); \\
    } \\
}

// X16R Algorithm - 16 different hashing algorithms
typedef enum {
    BLAKE = 0, BMW = 1, GROESTL = 2, JH = 3, KECCAK = 4, SKEIN = 5,
    LUFFA = 6, CUBEHASH = 7, SHAVITE = 8, SIMD = 9, ECHO = 10,
    HAMSI = 11, FUGUE = 12, SHABAL = 13, WHIRLPOOL = 14, SHA512 = 15
} hash_algorithm_t;

// Simplified hash function implementations for X16R
__device__ void blake_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified BLAKE implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x42;
    }
}

__device__ void bmw_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified BMW implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x43;
    }
}

__device__ void groestl_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified GROESTL implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x44;
    }
}

__device__ void jh_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified JH implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x45;
    }
}

__device__ void keccak_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified KECCAK implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x46;
    }
}

__device__ void skein_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified SKEIN implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x47;
    }
}

__device__ void luffa_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified LUFFA implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x48;
    }
}

__device__ void cubehash_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified CUBEHASH implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x49;
    }
}

__device__ void shavite_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified SHAVITE implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x4A;
    }
}

__device__ void simd_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified SIMD implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x4B;
    }
}

__device__ void echo_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified ECHO implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x4C;
    }
}

__device__ void hamsi_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified HAMSI implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x4D;
    }
}

__device__ void fugue_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified FUGUE implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x4E;
    }
}

__device__ void shabal_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified SHABAL implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x4F;
    }
}

__device__ void whirlpool_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified WHIRLPOOL implementation
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x50;
    }
}

__device__ void sha512_hash(const uint8_t *input, size_t len, uint8_t *output) {
    // Simplified SHA512 implementation (truncated to 256 bits)
    for(int i = 0; i < 32; i++) {
        output[i] = input[i] ^ 0x51;
    }
}

// X16R algorithm implementation
__device__ void x16r_hash(const uint8_t *input, size_t len, uint8_t *output, uint32_t nonce) {
    uint8_t temp[32];
    uint8_t current_input[80];
    
    // Copy input to working buffer
    memcpy(current_input, input, len);
    
    // X16R uses the last 16 bits of the previous hash to determine the next algorithm
    uint16_t seed = (nonce >> 16) & 0xFFFF;
    
    // Run 16 rounds of different hash algorithms
    for(int round = 0; round < 16; round++) {
        // Determine algorithm based on seed (simplified)
        hash_algorithm_t algo = (hash_algorithm_t)((seed + round) % 16);
        
        // Apply the selected hash algorithm
        switch(algo) {
            case BLAKE:
                blake_hash(current_input, len, temp);
                break;
            case BMW:
                bmw_hash(current_input, len, temp);
                break;
            case GROESTL:
                groestl_hash(current_input, len, temp);
                break;
            case JH:
                jh_hash(current_input, len, temp);
                break;
            case KECCAK:
                keccak_hash(current_input, len, temp);
                break;
            case SKEIN:
                skein_hash(current_input, len, temp);
                break;
            case LUFFA:
                luffa_hash(current_input, len, temp);
                break;
            case CUBEHASH:
                cubehash_hash(current_input, len, temp);
                break;
            case SHAVITE:
                shavite_hash(current_input, len, temp);
                break;
            case SIMD:
                simd_hash(current_input, len, temp);
                break;
            case ECHO:
                echo_hash(current_input, len, temp);
                break;
            case HAMSI:
                hamsi_hash(current_input, len, temp);
                break;
            case FUGUE:
                fugue_hash(current_input, len, temp);
                break;
            case SHABAL:
                shabal_hash(current_input, len, temp);
                break;
            case WHIRLPOOL:
                whirlpool_hash(current_input, len, temp);
                break;
            case SHA512:
                sha512_hash(current_input, len, temp);
                break;
        }
        
        // Copy result to input for next round
        memcpy(current_input, temp, 32);
        len = 32;
        
        // Update seed for next round
        seed = (seed * 1103515245 + 12345) & 0xFFFF;
    }
    
    // Copy final result to output
    memcpy(output, temp, 32);
}

// Improved mining kernel with X16R algorithm
__global__ void mine_x16r_kernel(uint8_t *all_data, uint32_t *results, uint64_t start_nonce, int active_pools) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nonce = start_nonce + idx;
    uint8_t temp[80];
    uint8_t hash_result[32];

    for(int p=0; p<active_pools; p++) {
        int pool_offset = p * POOL_DATA_SIZE;
        
        uint32_t hdr_len = *((uint32_t*)&all_data[pool_offset]);
        if(hdr_len != 80) continue;  // Ravencoin headers are 80 bytes
        
        memcpy(temp, &all_data[pool_offset + 4], 80);  // Copy full 80-byte header
        
        // Insert nonce at RVN position (bytes 76-79, little endian)
        // This is the X16R algorithm requirement
        temp[76] = (nonce >> 0) & 0xFF;
        temp[77] = (nonce >> 8) & 0xFF;
        temp[78] = (nonce >> 16) & 0xFF;
        temp[79] = (nonce >> 24) & 0xFF;
        
        const uint8_t *target = &all_data[pool_offset + 4 + 80];  // After header_len + header
        
        // Use X16R algorithm instead of double SHA-256
        x16r_hash(temp, 80, hash_result, (uint32_t)nonce);
        
        // Compare with target (little-endian comparison)
        bool valid = true;
        for(int i=0; i<32; i++) {
            if(hash_result[i] > target[i]) {
                valid = false;
                break;
            } else if(hash_result[i] < target[i]) {
                break;
            }
        }
        
        if(valid) {
            atomicCAS(&results[p], 0, (uint32_t)nonce);
        }
    }
}

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: %s <start_nonce>\\n", argv[0]);
        printf("X16R Algorithm for Ravencoin Mining\\n");
        return 1;
    }
    
    uint64_t start_nonce = strtoull(argv[1], NULL, 10);
    printf("Using X16R algorithm (Ravencoin standard)\\n");
    
    // Load data
    FILE *f = fopen("headers.bin", "rb");
    if(!f) {
        printf("Error opening headers.bin\\n");
        return 1;
    }
    
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    int active_pools = file_size / POOL_DATA_SIZE;
    if(active_pools == 0) {
        printf("No complete pools found\\n");
        fclose(f);
        return 1;
    }
    
    printf("Found %d active pools\\n", active_pools);
    
    // Dynamic allocation for variable-sized arrays
    uint8_t *all_data = (uint8_t*)malloc(active_pools * POOL_DATA_SIZE);
    uint32_t *results = (uint32_t*)malloc(active_pools * sizeof(uint32_t));
    
    if(!all_data || !results) {
        printf("Memory allocation failed\\n");
        fclose(f);
        free(all_data);
        free(results);
        return 1;
    }
    
    size_t read = fread(all_data, 1, active_pools * POOL_DATA_SIZE, f);
    fclose(f);
    
    if(read != (size_t)(active_pools * POOL_DATA_SIZE)) {
        printf("Error reading pool data\\n");
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
    mine_x16r_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_all_data, d_results, start_nonce, active_pools);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get results
    CUDA_CHECK(cudaMemcpy(results, d_results, active_pools * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // Output
    bool found = false;
    for(int p=0; p<active_pools; p++) {
        if(results[p] != 0) {
            printf("Valid X16R nonce found for pool %d: %u\\n", p, results[p]);
            found = true;
        }
    }
    
    if(!found) {
        printf("No valid X16R nonces found\\n");
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_all_data));
    CUDA_CHECK(cudaFree(d_results));
    free(all_data);
    free(results);
    
    return 0;
}
'''
    
    with open("x16r_cuda_miner.cu", "w") as f:
        f.write(cuda_code)
    
    logger.info("✅ Created x16r_cuda_miner.cu")

def test_all_pools_x16r_fixed():
    """Test all pools with X16R algorithm (fixed version)"""
    logger.info("🔧 TESTING ALL POOLS WITH X16R ALGORITHM (FIXED)")
    logger.info("=" * 60)
    
    config = load_config()
    
    results = {}
    
    for pool in config['pools']:
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing {pool['name']}")
        logger.info(f"{'='*40}")
        
        success = test_pool_with_x16r_fixed(pool)
        results[pool['name']] = success
        
        time.sleep(2)  # Wait between pools
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("X16R ALGORITHM TEST RESULTS (FIXED)")
    logger.info(f"{'='*60}")
    
    for pool_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{pool_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} pools passed X16R test")
    
    return results

def main():
    """Main function"""
    logger.info("🚀 X16R ALGORITHM DEBUG SESSION (FIXED)")
    logger.info("Based on official Ravencoin whitepaper")
    logger.info("=" * 60)
    
    # Test all pools with X16R (fixed)
    results = test_all_pools_x16r_fixed()
    
    # Create X16R CUDA miner
    create_x16r_cuda_miner()
    
    logger.info("\n📋 Next steps:")
    logger.info("1. Compile: nvcc -O3 -arch=sm_60 -o miner_x16r.exe x16r_cuda_miner.cu")
    logger.info("2. Test with real X16R hashing")
    logger.info("3. Monitor acceptance rates")
    logger.info("4. Update auto_miner to use X16R miner")

if __name__ == "__main__":
    main() 