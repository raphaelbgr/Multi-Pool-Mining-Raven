# BMAD Integration Guide for XMRig Multi-Pool KawPow Mining

## Overview

This guide explains how to integrate BMAD (Bidirectional Multi-Algorithm DAG) with XMRig to enable efficient multi-pool KawPow mining for RavenCoin.

## Prerequisites

- BMAD library built successfully
- XMRig source code
- CUDA 12.9 development environment
- Visual Studio 2022

## Step 1: Build BMAD Library

```bash
cd bmad-dev
build.bat
```

Verify the build produces:
- `build/lib/Release/bmad_kawpow_multi.dll`
- `build/include/bmad/` headers

## Step 2: Integrate BMAD with XMRig

### 2.1 Update XMRig CMakeLists.txt

Add BMAD to XMRig's build system:

```cmake
# In xmrig/CMakeLists.txt
if (WITH_CUDA)
    # Add BMAD library
    add_library(bmad_kawpow_multi SHARED IMPORTED)
    set_target_properties(bmad_kawpow_multi PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../bmad-dev/build/lib/Release/bmad_kawpow_multi.dll"
        IMPORTED_IMPLIB "${CMAKE_SOURCE_DIR}/../bmad-dev/build/lib/Release/bmad_kawpow_multi.lib"
    )
    
    # Add BMAD include directory
    include_directories("${CMAKE_SOURCE_DIR}/../bmad-dev/build/include")
    
    # Link BMAD to XMRig
    target_link_libraries(xmrig bmad_kawpow_multi)
endif()
```

### 2.2 Create BMAD KawPow Runner

Create `xmrig/src/backend/cuda/runners/CudaKawPowMultiRunner.h`:

```cpp
#ifndef XMRIG_CUDAKAWPOWMULTIRUNNER_H
#define XMRIG_CUDAKAWPOWMULTIRUNNER_H

#include "backend/cuda/runners/CudaBaseRunner.h"
#include "bmad_kawpow_multi.h"

namespace xmrig {

class CudaKawPowMultiRunner : public CudaBaseRunner
{
public:
    CudaKawPowMultiRunner(size_t index, const CudaLaunchData &data);

protected:
    bool run(uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce) override;
    bool set(const Job &job, uint8_t *blob) override;
    size_t processedHashes() const override { return intensity() - m_skippedHashes; }
    void jobEarlyNotification(const Job&) override;

private:
    std::unique_ptr<BMAD::KawPowMulti> m_bmad_miner;
    std::vector<BMAD::MultiPoolJob> m_pool_jobs;
    uint32_t m_skippedHashes = 0;
};

} // namespace xmrig

#endif // XMRIG_CUDAKAWPOWMULTIRUNNER_H
```

### 2.3 Implement BMAD KawPow Runner

Create `xmrig/src/backend/cuda/runners/CudaKawPowMultiRunner.cpp`:

```cpp
#include "backend/cuda/runners/CudaKawPowMultiRunner.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Chrono.h"
#include "crypto/kawpow/KPCache.h"
#include "crypto/kawpow/KPHash.h"

xmrig::CudaKawPowMultiRunner::CudaKawPowMultiRunner(size_t index, const CudaLaunchData &data) :
    CudaBaseRunner(index, data)
{
    // Initialize BMAD configuration
    BMAD::BMADConfig config;
    config.max_pools = 10;
    config.batch_size = 1024;
    config.memory_alignment = 4096;
    config.use_pinned_memory = true;
    config.enable_profiling = false;
    
    // Create BMAD miner
    m_bmad_miner = std::make_unique<BMAD::KawPowMulti>();
    if (!m_bmad_miner->initialize(config, data.device.index())) {
        LOG_ERR("%s " YELLOW("BMAD") RED(" failed to initialize"), Tags::nvidia());
    }
}

bool xmrig::CudaKawPowMultiRunner::run(uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce)
{
    if (!m_bmad_miner) {
        return false;
    }
    
    // Convert XMRig jobs to BMAD jobs
    std::vector<BMAD::MultiPoolJob> bmad_jobs;
    for (const auto& job : m_pool_jobs) {
        BMAD::MultiPoolJob bmad_job;
        memcpy(bmad_job.blob, job.blob, 40);
        bmad_job.target = job.target;
        bmad_job.height = job.height;
        bmad_job.pool_id = job.pool_id;
        bmad_job.active = job.active;
        bmad_jobs.push_back(bmad_job);
    }
    
    // Set jobs in BMAD miner
    if (!m_bmad_miner->setJobs(bmad_jobs)) {
        return false;
    }
    
    // Mine with BMAD
    std::vector<BMAD::MultiPoolResult> results;
    if (!m_bmad_miner->mine(startNonce, intensity(), results)) {
        return false;
    }
    
    // Convert results back to XMRig format
    *rescount = 0;
    for (const auto& result : results) {
        if (result.valid && *rescount < 16) {
            resnonce[*rescount] = result.nonce;
            (*rescount)++;
        }
    }
    
    m_skippedHashes = m_bmad_miner->getSkippedHashes();
    return true;
}

bool xmrig::CudaKawPowMultiRunner::set(const Job &job, uint8_t *blob)
{
    if (!CudaBaseRunner::set(job, blob)) {
        return false;
    }
    
    // Update pool jobs
    // This would be called from MultiPoolStrategy when new jobs arrive
    return true;
}

void xmrig::CudaKawPowMultiRunner::jobEarlyNotification(const Job&)
{
    if (m_bmad_miner) {
        m_bmad_miner->cleanup();
    }
}
```

### 2.4 Update MultiPoolStrategy

Modify `xmrig/src/base/net/stratum/strategies/MultiPoolStrategy.cpp`:

```cpp
// Add BMAD include
#include "bmad_kawpow_multi.h"

// In distributeJobResult method
void MultiPoolStrategy::distributeJobResult(const JobResult& result)
{
    // Use BMAD for multi-pool validation
    std::vector<BMAD::MultiPoolResult> bmad_results;
    
    // Convert current jobs to BMAD format
    std::vector<BMAD::MultiPoolJob> bmad_jobs;
    for (const auto& poolConn : m_poolConnections) {
        if (poolConn.currentJob.solved) continue;
        
        BMAD::MultiPoolJob job;
        memcpy(job.blob, poolConn.currentJob.job.blob(), 40);
        job.target = poolConn.currentJob.job.target();
        job.height = poolConn.currentJob.job.height();
        job.pool_id = poolConn.pool_id;
        job.active = true;
        bmad_jobs.push_back(job);
    }
    
    // Use BMAD to validate against all pools
    if (m_bmad_miner && !bmad_jobs.empty()) {
        m_bmad_miner->setJobs(bmad_jobs);
        m_bmad_miner->mine(result.nonce, 1, bmad_results);
        
        // Process results
        for (const auto& bmad_result : bmad_results) {
            if (bmad_result.valid) {
                // Submit share to corresponding pool
                submitShare(bmad_result.pool_id, bmad_result.nonce);
            }
        }
    }
}
```

## Step 3: Configure Multi-Pool Mining

### 3.1 Update XMRig Config

Modify your `config.json`:

```json
{
    "algo": "kawpow",
    "cuda": {
        "enabled": true,
        "loader": "bmad_kawpow_multi.dll",
        "kawpow": [
            { "index": 0 }
        ]
    },
    "pools": [
        {
            "url": "stratum+tcp://pool1.example.com:3333",
            "user": "your_wallet",
            "pass": "x",
            "algo": "kawpow"
        },
        {
            "url": "stratum+tcp://pool2.example.com:3333",
            "user": "your_wallet",
            "pass": "x",
            "algo": "kawpow"
        }
    ]
}
```

### 3.2 BMAD Configuration

Create `bmad_config.json` in XMRig directory:

```json
{
    "bmad_config": {
        "max_pools": 10,
        "batch_size": 1024,
        "memory_alignment": 4096,
        "use_pinned_memory": true
    },
    "performance_config": {
        "enable_multi_pool_mining": true,
        "enable_adaptive_batching": true
    }
}
```

## Step 4: Build and Test

### 4.1 Build XMRig with BMAD

```bash
cd xmrig
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DWITH_CUDA=ON
cmake --build . --config Release
```

### 4.2 Test Multi-Pool Mining

```bash
cd Release
./xmrig.exe --config=../../config.json --donate-level=0
```

## Performance Monitoring

### 4.1 Enable BMAD Profiling

```json
{
    "performance_config": {
        "enable_profiling": true,
        "log_level": "debug"
    }
}
```

### 4.2 Monitor Output

Look for BMAD-specific log messages:
- Multi-pool job processing
- Share distribution across pools
- Memory usage statistics
- Performance metrics

## Troubleshooting

### Common Issues

1. **BMAD Library Not Found**:
   - Verify `bmad_kawpow_multi.dll` is in XMRig directory
   - Check library dependencies

2. **CUDA Memory Errors**:
   - Reduce `max_pools` in BMAD config
   - Adjust `batch_size` for your GPU

3. **Build Errors**:
   - Ensure CUDA 12.9 is properly installed
   - Check Visual Studio C++ tools

### Debug Steps

1. **Test BMAD Library**:
   ```bash
   cd bmad-dev
   ./test_bmad.exe
   ```

2. **Verify CUDA Setup**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **Check XMRig Integration**:
   - Review log output for BMAD messages
   - Monitor share submission across pools

## Performance Optimization

### Memory Management

- **Pinned Memory**: Faster host-device transfers
- **Memory Alignment**: Optimized for CUDA access patterns
- **DAG Caching**: Efficient DAG storage

### Kernel Optimization

- **Warp-Level Primitives**: Efficient thread cooperation
- **Shared Memory**: Optimized cache access
- **Loop Unrolling**: Reduced loop overhead

### Multi-Pool Efficiency

- **Single Nonce, Multiple Pools**: One computation validates against all pools
- **Batch Processing**: Process multiple nonces simultaneously
- **Adaptive Batching**: Dynamic batch size based on GPU capability

## Expected Performance Gains

With BMAD multi-pool mining, you should see:

- **Increased Efficiency**: Single CUDA iteration validates against multiple pools
- **Higher Hashrate**: More shares found per computation
- **Better Pool Distribution**: Automatic load balancing across pools
- **Reduced Latency**: Faster share submission to multiple pools

## Next Steps

1. **Fine-tune Configuration**: Adjust BMAD parameters for your GPU
2. **Monitor Performance**: Track hashrate and share distribution
3. **Scale Up**: Add more pools to maximize efficiency
4. **Optimize Further**: Profile and optimize kernel performance