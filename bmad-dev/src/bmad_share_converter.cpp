#include "../include/bmad_share_converter.h"
#include "../src/bmad_kawpow_algorithm.h"
#include <iostream>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace BMAD {

ShareConverter::ShareConverter() {
    std::cout << "🔄 Share Converter initialized" << std::endl;
}

ShareConverter::~ShareConverter() {
    std::cout << "🔄 Share Converter destroyed" << std::endl;
}

bool ShareConverter::convertBMADShareToXMRig(
    const MultiPoolJob& bmad_job, 
    uint32_t nonce, 
    uint32_t pool_id,
    xmrig::JobResult& xmrig_result) 
{
    try {
        std::cout << "🔄 Converting BMAD share to XMRig format (pool " << pool_id << ", nonce " << nonce << ")" << std::endl;
        
        // Set basic job information
        xmrig_result.jobId = std::to_string(bmad_job.height);
        xmrig_result.nonce = nonce;
        xmrig_result.poolId = static_cast<int>(pool_id);
        
        // Copy job blob
        std::memcpy(xmrig_result.blob, bmad_job.blob, sizeof(bmad_job.blob));
        
        // Calculate hash for this nonce
        uint8_t hash[32];
        if (!calculateHashForNonce(bmad_job, nonce, hash)) {
            std::cerr << "❌ Failed to calculate hash for nonce " << nonce << std::endl;
            return false;
        }
        
        // Convert hash to hex string
        std::string hash_hex = bytesToHex(hash, 32);
        xmrig_result.result = hash_hex;
        
        // Set difficulty and target
        xmrig_result.diff = bmad_job.target;
        xmrig_result.actualDiff = bmad_job.target; // Use target as actual diff for now
        
        // Set timestamp
        xmrig_result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::cout << "✅ Share converted successfully" << std::endl;
        std::cout << "  Pool ID: " << pool_id << std::endl;
        std::cout << "  Nonce: " << nonce << std::endl;
        std::cout << "  Hash: " << hash_hex.substr(0, 16) << "..." << std::endl;
        std::cout << "  Target: 0x" << std::hex << bmad_job.target << std::dec << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception during share conversion: " << e.what() << std::endl;
        return false;
    }
}

bool ShareConverter::calculateHashForNonce(
    const MultiPoolJob& job, 
    uint32_t nonce, 
    uint8_t* hash) 
{
    try {
        // Create a copy of the job blob with the nonce
        uint8_t job_with_nonce[sizeof(job.blob)];
        std::memcpy(job_with_nonce, job.blob, sizeof(job.blob));
        
        // Insert nonce into the correct position (typically bytes 76-79 for KawPow)
        // This is the standard position for nonce in KawPow job blobs
        job_with_nonce[76] = (nonce >> 0) & 0xFF;
        job_with_nonce[77] = (nonce >> 8) & 0xFF;
        job_with_nonce[78] = (nonce >> 16) & 0xFF;
        job_with_nonce[79] = (nonce >> 24) & 0xFF;
        
        // Calculate KawPow hash - using the correct signature
        // Note: This is a simplified version - in a real implementation, 
        // we would need the DAG and proper CUDA context
        uint64_t hash_result = KawPowAlgorithm::calculateHash(
            job_with_nonce, 
            nonce, 
            nullptr,  // DAG would be passed here in real implementation
            0         // DAG size would be passed here in real implementation
        );
        
        // Convert uint64_t hash to uint8_t array
        for (int i = 0; i < 8; ++i) {
            hash[i] = (hash_result >> (i * 8)) & 0xFF;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception during hash calculation: " << e.what() << std::endl;
        return false;
    }
}

std::string ShareConverter::bytesToHex(const uint8_t* bytes, size_t length) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    
    for (size_t i = 0; i < length; ++i) {
        ss << std::setw(2) << static_cast<int>(bytes[i]);
    }
    
    return ss.str();
}

bool ShareConverter::validateShare(const xmrig::JobResult& result, uint32_t target) {
    try {
        // Convert hex result back to bytes
        std::vector<uint8_t> hash_bytes = hexToBytes(result.result);
        
        if (hash_bytes.size() != 32) {
            std::cerr << "❌ Invalid hash length: " << hash_bytes.size() << std::endl;
            return false;
        }
        
        // Check if hash meets target difficulty
        // For KawPow, we check if the hash is less than the target
        uint64_t hash_value = 0;
        for (int i = 0; i < 8; ++i) {
            hash_value |= (static_cast<uint64_t>(hash_bytes[7-i]) << (i * 8));
        }
        
        bool is_valid = hash_value < target;
        
        std::cout << "🔍 Share validation:" << std::endl;
        std::cout << "  Hash value: 0x" << std::hex << hash_value << std::dec << std::endl;
        std::cout << "  Target: 0x" << std::hex << target << std::dec << std::endl;
        std::cout << "  Valid: " << (is_valid ? "✅ YES" : "❌ NO") << std::endl;
        
        return is_valid;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception during share validation: " << e.what() << std::endl;
        return false;
    }
}

std::vector<uint8_t> ShareConverter::hexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_string = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoi(byte_string, nullptr, 16));
        bytes.push_back(byte);
    }
    
    return bytes;
}

} // namespace BMAD