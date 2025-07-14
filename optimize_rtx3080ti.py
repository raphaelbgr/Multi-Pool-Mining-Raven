#!/usr/bin/env python3
"""
Otimização da RTX 3080 Ti para mineração
"""

import subprocess
import os
import time

def check_gpu_usage():
    """Verifica uso atual das GPUs"""
    
    print("🔍 Verificando uso das GPUs...")
    print("=" * 50)
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_id = parts[0]
                    name = parts[1]
                    utilization = parts[2]
                    mem_used = parts[3]
                    mem_total = parts[4]
                    
                    print(f"GPU {gpu_id}: {name}")
                    print(f"  Utilização: {utilization}%")
                    print(f"  Memória: {mem_used}/{mem_total} MB")
                    
                    if int(utilization) > 80:
                        print(f"  ⚠️  ALTA UTILIZAÇÃO - Pode afetar performance")
                    print()
        else:
            print("❌ Erro ao verificar GPUs")
            
    except Exception as e:
        print(f"❌ Erro: {str(e)}")

def optimize_for_mining():
    """Otimiza configurações para mineração"""
    
    print("⚡ Otimizando para mineração...")
    print("=" * 50)
    
    # Configurações recomendadas para RTX 3080 Ti
    optimizations = [
        "set CUDA_VISIBLE_DEVICES=0",
        "set CUDA_LAUNCH_BLOCKING=0", 
        "set CUDA_CACHE_DISABLE=0",
        "set CUDA_FORCE_PTX_JIT=0"
    ]
    
    print("📋 Configurações recomendadas:")
    for opt in optimizations:
        print(f"  {opt}")
    
    # Criar script otimizado
    optimized_script = """@echo off
echo ========================================
echo    Multi-Pool Miner - RTX 3080 Ti OTIMIZADO
echo ========================================
echo.

echo Configurando ambiente otimizado...
set CUDA_VISIBLE_DEVICES=0
set CUDA_LAUNCH_BLOCKING=0
set CUDA_CACHE_DISABLE=0
set CUDA_FORCE_PTX_JIT=0

echo.
echo IMPORTANTE: Pare o BitCrack para melhor performance!
echo.
pause

echo.
echo Iniciando minerador otimizado...
echo.

python auto_miner.py

echo.
echo Minerador parado.
pause
"""
    
    with open("start_miner_optimized.bat", "w") as f:
        f.write(optimized_script)
    
    print("✅ Script otimizado criado: start_miner_optimized.bat")

def test_optimized_performance():
    """Testa performance com configurações otimizadas"""
    
    print("\n🔧 Testando performance otimizada...")
    print("=" * 50)
    
    # Aplicar configurações otimizadas
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['CUDA_FORCE_PTX_JIT'] = '0'
    
    # Testar hashrate
    start_time = time.time()
    
    result = subprocess.run(
        ["./miner.exe", "0"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    if result.returncode == 0:
        nonces_tested = 262144 * 1024
        hashrate = nonces_tested / duration
        hashrate_gh = hashrate / 1_000_000_000
        
        print(f"✅ Hashrate otimizado: {hashrate_gh:.2f} GH/s")
        print(f"⏱️  Tempo: {duration:.2f} segundos")
        
        if hashrate_gh > 2.0:
            print("🎉 Performance EXCELENTE!")
        elif hashrate_gh > 1.5:
            print("✅ Performance BOA")
        else:
            print("⚠️ Performance ainda baixa")
    else:
        print("❌ Erro no teste")

def create_dual_mining_script():
    """Cria script para mineração em ambas GPUs (se possível)"""
    
    print("\n🔄 Criando script para mineração dual...")
    print("=" * 50)
    
    dual_script = """@echo off
echo ========================================
echo    Multi-Pool Miner - DUAL GPU
echo ========================================
echo.

echo Configurando para ambas GPUs...
echo RTX 3080 Ti: GPU 0
echo GTX 1060: GPU 1
echo.

echo IMPORTANTE: Pare o BitCrack para melhor performance!
echo.
pause

echo.
echo Iniciando minerador em ambas GPUs...
echo.

REM Iniciar minerador na RTX 3080 Ti
start "RTX 3080 Ti Miner" cmd /c "set CUDA_VISIBLE_DEVICES=0 && python auto_miner.py"

REM Aguardar um pouco
timeout /t 5 /nobreak >nul

REM Iniciar minerador na GTX 1060 (se disponível)
start "GTX 1060 Miner" cmd /c "set CUDA_VISIBLE_DEVICES=1 && python auto_miner.py"

echo.
echo Ambos mineradores iniciados!
echo Pressione qualquer tecla para parar...
pause

taskkill /f /im python.exe
echo Mineradores parados.
pause
"""
    
    with open("start_dual_mining.bat", "w") as f:
        f.write(dual_script)
    
    print("✅ Script dual criado: start_dual_mining.bat")

if __name__ == "__main__":
    print("🎯 Otimizador RTX 3080 Ti")
    print("=" * 50)
    
    # Verificar uso atual
    check_gpu_usage()
    
    # Otimizar configurações
    optimize_for_mining()
    
    # Testar performance otimizada
    test_optimized_performance()
    
    # Criar script dual (opcional)
    print("\n" + "=" * 50)
    response = input("Deseja criar script para mineração dual? (s/n): ")
    if response.lower() == 's':
        create_dual_mining_script()
    
    print("\n✅ Otimização concluída!")
    print("💡 Scripts disponíveis:")
    print("  - start_miner_optimized.bat (RTX 3080 Ti otimizado)")
    print("  - start_dual_mining.bat (ambas GPUs)")
    print("  - start_miner_rtx3080ti.bat (básico)") 