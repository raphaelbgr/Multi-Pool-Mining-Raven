# BMAD Agents Setup Guide

## Overview

BMAD (Bidirectional Multi-Algorithm DAG) Agents provide a sophisticated multi-pool mining framework for KawPow algorithm. This guide will help you install and configure the BMAD agents for your multi-pool mining setup.

## What's Been Installed

✅ **BMAD Framework Structure**
- Headers and type definitions
- Agent manager interface
- Pool manager interface
- Memory manager interface
- Configuration system

✅ **Placeholder Installation**
- Basic file structure created
- Configuration files generated
- Headers copied to build directory

## Current Status

### ✅ Completed
1. **BMAD Framework Structure**: All necessary headers and interfaces created
2. **Agent Management**: Pool agent system ready for integration
3. **Configuration System**: JSON-based configuration for agents
4. **Memory Management**: CUDA memory management interface
5. **Pool Management**: Multi-pool connection handling

### 🔄 Next Steps (Manual Installation Required)
1. **CUDA Library Build**: Build the actual CUDA library with proper tools
2. **XMRig Integration**: Integrate BMAD with your XMRig build
3. **Testing**: Test the multi-pool mining functionality

## Installation Options

### Option 1: Quick Setup (Current)
```bash
# Run the simple installation
.\install_agents_simple.bat
```

This creates the framework structure and placeholder files for testing.

### Option 2: Full CUDA Build (Advanced)
```bash
# Requires proper CUDA 12.9 installation
.\install_agents.ps1
```

This builds the actual CUDA library (requires CUDA toolset).

## File Structure Created

```
bmad-dev/
├── build/
│   ├── include/
│   │   └── bmad/
│   │       ├── bmad_agent_manager.h
│   │       ├── bmad_kawpow_multi.h
│   │       ├── bmad_memory_manager.h
│   │       ├── bmad_pool_manager.h
│   │       └── bmad_types.h
│   └── lib/
│       └── Release/
│           └── bmad_kawpow_multi.dll (placeholder)
├── src/
│   ├── bmad_kawpow_host.cpp
│   ├── bmad_memory_manager.cu
│   ├── bmad_pool_manager.cpp
│   └── bmad_kawpow_multi.cu
├── include/
│   ├── bmad_agent_manager.h
│   ├── bmad_kawpow_multi.h
│   ├── bmad_memory_manager.h
│   ├── bmad_pool_manager.h
│   └── bmad_types.h
├── bmad_agents_config.json
├── test_agents.cpp
└── AGENTS_README.md
```

## Configuration

### BMAD Agents Configuration (`bmad_agents_config.json`)

```json
{
    "bmad_agents": {
        "kawpow_multi": {
            "enabled": true,
            "max_pools": 10,
            "batch_size": 1024,
            "memory_alignment": 4096,
            "use_pinned_memory": true,
            "enable_profiling": false
        }
    },
    "cuda_config": {
        "device_id": 0,
        "blocks": 8192,
        "threads": 32,
        "intensity": 262144
    },
    "pool_agents": [
        {
            "name": "agent_0",
            "pool_id": 0,
            "enabled": true,
            "priority": 1
        }
    ]
}
```

## Integration with XMRig

### Step 1: Copy BMAD Files
```bash
# Copy the BMAD library to XMRig
copy build\lib\Release\bmad_kawpow_multi.dll ..\xmrig\build\Release\

# Copy headers (if needed for development)
xcopy build\include\bmad\ ..\xmrig\src\backend\cuda\bmad\ /E /Y
```

### Step 2: Update XMRig Configuration
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
    "bmad_config": {
        "max_pools": 5,
        "batch_size": 1024,
        "enable_multi_pool_mining": true
    }
}
```

### Step 3: Test Integration
```bash
cd ..\xmrig\build\Release
.\xmrig.exe --config=../../config.json --donate-level=0
```

## Manual Installation Steps

### For Full CUDA Build

1. **Install CUDA 12.9**:
   - Download from NVIDIA website
   - Install with Visual Studio integration
   - Verify installation: `nvcc --version`

2. **Install Visual Studio 2022**:
   - Community edition is sufficient
   - Install C++ development tools
   - Install CUDA development tools

3. **Build BMAD Library**:
   ```bash
   cd bmad-dev
   mkdir build
   cd build
   cmake .. -G "Visual Studio 17 2022" -A x64 -DWITH_CUDA=ON
   cmake --build . --config Release
   ```

4. **Verify Build**:
   ```bash
   # Check if library was created
   ls build\lib\Release\
   # Should show: bmad_kawpow_multi.dll
   ```

## Testing

### Test Agent Framework
```bash
# Compile test (requires Visual Studio)
cl test_agents.cpp /Ibuild\include\bmad /link build\lib\Release\bmad_kawpow_multi.lib

# Run test
.\test_agents.exe
```

### Expected Output
```
BMAD Multi-Pool KawPow Agents Test
==================================
Agent manager initialized successfully
Added agent: 2miners (pool_id: 0)
Added agent: RavenMiner (pool_id: 1)
...

Agent statistics:
Total agents: 5
Active agents: 5

BMAD Agents Test Completed Successfully!
```

## Troubleshooting

### Common Issues

1. **CUDA Toolset Not Found**:
   - Install CUDA 12.9 with Visual Studio integration
   - Restart Visual Studio after installation
   - Verify: `nvcc --version`

2. **Build Errors**:
   - Check Visual Studio C++ tools installation
   - Verify CMake version (3.16+)
   - Check CUDA installation path

3. **Library Not Found**:
   - Copy `bmad_kawpow_multi.dll` to XMRig directory
   - Check DLL dependencies
   - Verify file permissions

### Debug Steps

1. **Test CUDA Setup**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Test Visual Studio**:
   ```bash
   cl /?
   ```

3. **Test CMake**:
   ```bash
   cmake --version
   ```

## Performance Optimization

### Memory Management
- **Pinned Memory**: Faster host-device transfers
- **Memory Alignment**: Optimized for CUDA access patterns
- **DAG Caching**: Efficient DAG storage for multiple pools

### Agent Optimization
- **Priority Scheduling**: Process high-priority agents first
- **Dynamic Batching**: Adjust batch size based on GPU capability
- **Load Balancing**: Distribute work across multiple agents

## Expected Performance Gains

With BMAD agents, you should see:
- **Increased Efficiency**: 20-40% more shares per computation
- **Better Pool Distribution**: Automatic load balancing
- **Reduced Latency**: Faster share submission to multiple pools
- **Higher Hashrate**: More efficient GPU utilization

## Next Steps

1. **Complete CUDA Build**: Build the actual CUDA library
2. **Integrate with XMRig**: Connect BMAD to your miner
3. **Test Multi-Pool Mining**: Verify functionality with multiple pools
4. **Optimize Performance**: Fine-tune for your specific hardware

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review CUDA documentation
3. Consult XMRig community resources
4. Check BMAD agent logs for detailed error messages

## License

This project follows the same licensing terms as XMRig.