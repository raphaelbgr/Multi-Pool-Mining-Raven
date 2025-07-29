#ifndef BMAD_SHARE_CONVERTER_H
#define BMAD_SHARE_CONVERTER_H

#include "bmad_types.h"
#include <string>
#include <vector>
#include <cstdint>

// Forward declarations
namespace BMAD {
    class KawPowAlgorithm;
}

// Forward declaration for XMRig JobResult
namespace xmrig {
    struct JobResult {
        std::string jobId;
        uint32_t nonce;
        int poolId;
        uint8_t blob[40];
        std::string result;
        uint32_t diff;
        uint32_t actualDiff;
        uint64_t timestamp;
    };
}

namespace BMAD {

// Share converter for converting BMAD shares to XMRig format
class ShareConverter {
public:
    ShareConverter();
    ~ShareConverter();
    
    // Convert BMAD share to XMRig JobResult format
    bool convertBMADShareToXMRig(
        const MultiPoolJob& bmad_job, 
        uint32_t nonce, 
        uint32_t pool_id,
        xmrig::JobResult& xmrig_result);
    
    // Calculate hash for a specific nonce
    bool calculateHashForNonce(
        const MultiPoolJob& job, 
        uint32_t nonce, 
        uint8_t* hash);
    
    // Validate share against target difficulty
    bool validateShare(const xmrig::JobResult& result, uint32_t target);
    
    // Utility functions
    std::string bytesToHex(const uint8_t* bytes, size_t length);
    std::vector<uint8_t> hexToBytes(const std::string& hex);

private:
    // Internal helper methods
    bool insertNonceIntoBlob(uint8_t* blob, uint32_t nonce);
    bool verifyShareDifficulty(const uint8_t* hash, uint32_t target);
};

} // namespace BMAD

#endif // BMAD_SHARE_CONVERTER_H