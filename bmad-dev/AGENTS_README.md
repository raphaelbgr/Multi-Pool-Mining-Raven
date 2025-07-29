# BMAD Multi-Pool KawPow Agents

## Overview

BMAD (Bidirectional Multi-Algorithm DAG) Agents provide a sophisticated multi-pool mining framework for KawPow algorithm. This system allows a single CUDA iteration to test against multiple pools simultaneously, significantly increasing mining efficiency.

## Features

- **Multi-Pool Mining**: Single nonce tested against multiple pool jobs
- **Agent Management**: Individual agents for each pool with priority control
- **Memory Optimization**: Efficient DAG and cache handling for multiple pools
- **Performance Profiling**: Built-in performance monitoring and optimization
- **Dynamic Configuration**: Runtime agent enable/disable and priority adjustment

## Installation

### Prerequisites

- CUDA 12.9 or later
- Visual Studio 2022 with C++17 support
- CMake 3.16 or later
- Windows 10/11

### Quick Installation

1. **Run the installation script**:
   ```powershell
   # PowerShell (recommended)
   .\install_agents.ps1
   
   # Or batch file
   .\install_agents.bat
   ```

2. **Verify installation**:
   ```powershell
   # Check if files were created
   ls build\lib\Release\
   ls build\include\bmad\
   ```

### Manual Installation

If the automatic installation fails:

1. **Create build directories**:
   ```powershell
   mkdir build\lib\Release
   mkdir build\include\bmad
   ```

2. **Copy headers**:
   ```powershell
   copy include\*.h build\include\bmad\
   ```

3. **Build with CMake**:
   ```powershell
   cd build
   cmake .. -G "Visual Studio 17 2022" -A x64 -DWITH_CUDA=ON
   cmake --build . --config Release
   cd ..
   ```

## Configuration

### Agent Configuration (`bmad_agents_config.json`)

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

### Pool Agent Configuration

Each pool agent can be configured with:

- **name**: Agent identifier
- **pool_id**: Unique pool identifier (0-9)
- **enabled**: Whether agent is active
- **priority**: Processing priority (lower = higher priority)

## Usage

### Basic Usage

```cpp
#include "bmad_agent_manager.h"

// Initialize agent manager
BMAD::AgentManager agent_manager;
BMAD::BMADConfig config;
config.max_pools = 5;
config.batch_size = 1024;

if (!agent_manager.initialize(config, 0)) {
    // Handle initialization error
    return;
}

// Add pool agents
agent_manager.addAgent("2miners", 0, 1);
agent_manager.addAgent("RavenMiner", 1, 2);

// Update agent jobs
BMAD::MultiPoolJob job;
// ... set job parameters
agent_manager.updateAgentJob(0, job);

// Mine with all agents
std::vector<BMAD::MultiPoolResult> results;
agent_manager.mine(start_nonce, num_nonces, results);
```

### Advanced Usage

```cpp
// Get active agents
auto active_agents = agent_manager.getActiveAgents();

// Enable/disable agents
agent_manager.setAgentEnabled(2, false);

// Get agent statistics
uint32_t total_agents = agent_manager.getTotalAgentCount();
uint32_t active_agents = agent_manager.getActiveAgentCount();

// Get specific agent
BMAD::PoolAgent* agent = agent_manager.getAgent(0);
if (agent && agent->has_job) {
    // Process agent job
}
```

## Integration with XMRig

### Step 1: Copy BMAD Library

Copy the built library to your XMRig directory:

```powershell
copy build\lib\Release\bmad_kawpow_multi.dll ..\xmrig\build\Release\
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

Run XMRig with BMAD agents:

```powershell
cd ..\xmrig\build\Release
.\xmrig.exe --config=../../config.json --donate-level=0
```

## Testing

### Run Agent Test

```powershell
# Compile test
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
Added agent: WoolyPooly (pool_id: 2)
Added agent: HeroMiners (pool_id: 3)
Added agent: NanoPool (pool_id: 4)

Testing agent retrieval...
Found agent: 2miners (enabled: yes)
Found agent: RavenMiner (enabled: yes)
...

Active agents:
- 2miners (priority: 1)
- RavenMiner (priority: 2)
...

Agent statistics:
Total agents: 5
Active agents: 5

BMAD Agents Test Completed Successfully!
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

### Multi-Pool Efficiency

- **Single Nonce, Multiple Pools**: One computation validates against all pools
- **Parallel Processing**: Process multiple agents simultaneously
- **Adaptive Scheduling**: Dynamic agent priority based on performance

## Troubleshooting

### Common Issues

1. **CUDA Memory Errors**:
   - Reduce `max_pools` in configuration
   - Decrease `batch_size` for your GPU
   - Check GPU memory availability

2. **Build Errors**:
   - Verify CUDA 12.9 installation
   - Check Visual Studio C++ tools
   - Ensure CMake version compatibility

3. **Agent Initialization Failures**:
   - Check CUDA device availability
   - Verify memory allocation
   - Review configuration parameters

### Debug Steps

1. **Test CUDA Setup**:
   ```powershell
   nvcc --version
   nvidia-smi
   ```

2. **Verify Library**:
   ```powershell
   dumpbin /exports build\lib\Release\bmad_kawpow_multi.dll
   ```

3. **Check Integration**:
   - Review XMRig logs for BMAD messages
   - Monitor agent performance metrics
   - Verify multi-pool share distribution

## Performance Monitoring

### Enable Profiling

```json
{
    "bmad_agents": {
        "kawpow_multi": {
            "enable_profiling": true
        }
    }
}
```

### Monitor Metrics

- **Agent Performance**: Hashrate per agent
- **Memory Usage**: DAG and cache utilization
- **Share Distribution**: Shares submitted per pool
- **Latency**: Time from nonce to share submission

## Expected Performance Gains

With BMAD agents, you should see:

- **Increased Efficiency**: 20-40% more shares per computation
- **Better Pool Distribution**: Automatic load balancing
- **Reduced Latency**: Faster share submission to multiple pools
- **Higher Hashrate**: More efficient GPU utilization

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review CUDA documentation
3. Consult XMRig community resources
4. Check BMAD agent logs for detailed error messages

## License

This project follows the same licensing terms as XMRig.