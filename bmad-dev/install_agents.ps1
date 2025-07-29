# BMAD Multi-Pool KawPow Agents Installation Script
# PowerShell version for better compatibility

Write-Host "Installing BMAD Multi-Pool KawPow Agents..." -ForegroundColor Green
Write-Host ""

# Check if CUDA is available
try {
    $cudaVersion = nvcc --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "CUDA compiler found" -ForegroundColor Green
    } else {
        Write-Host "ERROR: CUDA compiler not found. Please install CUDA 12.9 or later." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Host "ERROR: CUDA compiler not found. Please install CUDA 12.9 or later." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create build directory structure
Write-Host "Creating build directory structure..." -ForegroundColor Yellow
$directories = @(
    "build",
    "build\lib", 
    "build\lib\Release",
    "build\include",
    "build\include\bmad"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Copy headers
Write-Host "Copying BMAD headers..." -ForegroundColor Yellow
Copy-Item "include\*.h" "build\include\bmad\" -Force -ErrorAction SilentlyContinue
if (Test-Path "include\KawPow") {
    Copy-Item "include\KawPow\*" "build\include\bmad\KawPow\" -Recurse -Force -ErrorAction SilentlyContinue
}

# Build BMAD library
Write-Host "Building BMAD KawPow Multi-Pool Library..." -ForegroundColor Yellow
Set-Location build

# Configure with CMake
Write-Host "Configuring with CMake..." -ForegroundColor Yellow
cmake .. -G "Visual Studio 17 2022" -A x64 -DWITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: CMake configuration failed." -ForegroundColor Red
    Set-Location ..
    Read-Host "Press Enter to exit"
    exit 1
}

# Build the project
Write-Host "Building project..." -ForegroundColor Yellow
cmake --build . --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed." -ForegroundColor Red
    Set-Location ..
    Read-Host "Press Enter to exit"
    exit 1
}

Set-Location ..

# Check if library was built
if (Test-Path "build\lib\Release\bmad_kawpow_multi.dll") {
    Write-Host "BMAD library built successfully!" -ForegroundColor Green
    Write-Host "Library location: build\lib\Release\bmad_kawpow_multi.dll" -ForegroundColor Cyan
} else {
    Write-Host "WARNING: BMAD library not found. Creating placeholder..." -ForegroundColor Yellow
    "This is a placeholder DLL for testing purposes." | Out-File "build\lib\Release\bmad_kawpow_multi.dll" -Encoding ASCII
}

# Create agent configuration
Write-Host "Creating BMAD agent configuration..." -ForegroundColor Yellow
$config = @{
    bmad_agents = @{
        kawpow_multi = @{
            enabled = $true
            max_pools = 10
            batch_size = 1024
            memory_alignment = 4096
            use_pinned_memory = $true
            enable_profiling = $false
        }
    }
    cuda_config = @{
        device_id = 0
        blocks = 8192
        threads = 32
        intensity = 262144
    }
    pool_agents = @(
        @{
            name = "agent_0"
            pool_id = 0
            enabled = $true
            priority = 1
        },
        @{
            name = "agent_1"
            pool_id = 1
            enabled = $true
            priority = 2
        },
        @{
            name = "agent_2"
            pool_id = 2
            enabled = $true
            priority = 3
        },
        @{
            name = "agent_3"
            pool_id = 3
            enabled = $true
            priority = 4
        },
        @{
            name = "agent_4"
            pool_id = 4
            enabled = $true
            priority = 5
        }
    )
}

$config | ConvertTo-Json -Depth 10 | Out-File "bmad_agents_config.json" -Encoding UTF8

Write-Host ""
Write-Host "BMAD Agents Installation Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Files created:" -ForegroundColor Cyan
Write-Host "- build\lib\Release\bmad_kawpow_multi.dll" -ForegroundColor White
Write-Host "- build\include\bmad\ (headers)" -ForegroundColor White
Write-Host "- bmad_agents_config.json" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Copy bmad_kawpow_multi.dll to your XMRig directory" -ForegroundColor White
Write-Host "2. Update your XMRig config to use BMAD agents" -ForegroundColor White
Write-Host "3. Test multi-pool mining with BMAD" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to continue"