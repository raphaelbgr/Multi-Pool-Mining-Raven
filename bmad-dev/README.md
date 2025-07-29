# BMAD Multi-Pool KawPow Mining Framework

## Overview

BMAD (Bidirectional Multi-Algorithm DAG) is a high-performance CUDA framework for multi-pool KawPow mining. This framework allows a single CUDA iteration to test against multiple pools simultaneously, significantly increasing mining efficiency.

## Key Features

- **Multi-Pool Mining**: Single CUDA iteration validates against multiple pools
- **Efficient Memory Management**: Optimized DAG and cache handling
- **Adaptive Batching**: Dynamic batch size optimization
- **Pool Management**: Automatic pool connection and job distribution
- **Performance Profiling**: Built-in performance monitoring

## Architecture

### Core Components

1. **BMADMemory**: Efficient memory management for DAG, cache, and job data
2. **KawPowMulti**: Multi-pool KawPow mining kernel
3. **PoolManager**: Pool connection and job management
4. **MemoryManager**: CUDA memory allocation and optimization

### Multi-Pool Flow

```
Single Nonce → Multiple Pool Jobs → Parallel Hash Computation → Share Distribution
```

## Building

### Prerequisites

- CUDA 12.9 or later
- Visual Studio 2022 with C++17 support
- CMake 3.16 or later

### Build Steps

1. **Clone and setup**:
   ```bash
   cd bmad-dev
   ```

2. **Build the library**:
   ```bash
   build.bat
   ```

3. **Verify build**:
   - Check `build/lib/Release/bmad_kawpow_multi.dll`
   - Check `build/include/bmad/` for headers

## Configuration

### BMAD Configuration (`bmad_config.json`)

```json
{
    "bmad_config": {
        "max_pools": 10,
        "batch_size": 1024,
        "memory_alignment": 4096,
        "use_pinned_memory": true
    },
    "cuda_config": {
        "device_id": 0,
        "blocks": 8192,
        "threads": 32,
        "intensity": 262144
    }
}
```

### Pool Configuration

Add your RavenCoin pools to the configuration:

```json
{
    "pools": [
        {
            "url": "stratum+tcp://your-pool.com:3333",
            "user": "your_wallet_address",
            "password": "x",
            "pool_id": 0,
            "active": true
        }
    ]
}
```

## Integration with XMRig

### Step 1: Replace KawPow Runner

Replace the existing `CudaKawPowRunner` with BMAD implementation:

```cpp
// In CudaKawPowRunner.cpp
bool xmrig::CudaKawPowRunner::run(uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce)
{
    // Use BMAD multi-pool kernel instead of single pool
    return BMAD::KawPowMulti::mine(startNonce, rescount, resnonce);
}
```

### Step 2: Update Job Distribution

Modify the job distribution to handle multiple pools:

```cpp
// In MultiPoolStrategy.cpp
void MultiPoolStrategy::distributeJobResult(const JobResult& result)
{
    // Use BMAD to validate against all pools simultaneously
    std::vector<MultiPoolResult> bmad_results;
    m_bmad_miner->mine(result.nonce, 1, bmad_results);
    
    for (const auto& bmad_result : bmad_results) {
        if (bmad_result.valid) {
            submitShare(bmad_result.pool_id, bmad_result.nonce);
        }
    }
}
```

## Performance Optimization

### Memory Management

- **Pinned Memory**: Faster host-device transfers
- **Memory Alignment**: Optimized for CUDA memory access patterns
- **DAG Caching**: Efficient DAG storage and retrieval

### Kernel Optimization

- **Warp-Level Primitives**: Efficient thread cooperation
- **Shared Memory**: Optimized cache access
- **Loop Unrolling**: Reduced loop overhead

### Multi-Pool Efficiency

- **Single Nonce, Multiple Pools**: One computation validates against all pools
- **Batch Processing**: Process multiple nonces simultaneously
- **Adaptive Batching**: Dynamic batch size based on GPU capability

## Troubleshooting

### Common Issues

1. **CUDA Memory Errors**:
   - Check GPU memory availability
   - Reduce batch size or number of pools

2. **Build Errors**:
   - Verify CUDA installation
   - Check Visual Studio C++ tools

3. **Performance Issues**:
   - Adjust blocks/threads configuration
   - Monitor GPU utilization

### Debugging

Enable profiling in configuration:

```json
{
    "performance_config": {
        "enable_profiling": true,
        "log_level": "debug"
    }
}
```

## License

This project is based on XMRig and follows the same licensing terms.

## Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review CUDA documentation
- Consult XMRig community resources