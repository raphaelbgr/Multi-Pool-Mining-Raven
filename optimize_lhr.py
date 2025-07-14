#!/usr/bin/env python3
"""
Otimização específica para RTX 3080 Ti LHR
"""

import subprocess
import os
import time

def check_lhr_status():
    """Verifica status da GPU LHR"""
    
    print("🔍 Verificando RTX 3080 Ti LHR...")
    print("=" * 50)
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu", 
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
                    mem_total = parts[2]
                    mem_free = parts[3]
                    utilization = parts[4]
                    
                    if "RTX 3080 Ti" in name:
                        print(f"🎯 RTX 3080 Ti LHR Detectada:")
                        print(f"  GPU ID: {gpu_id}")
                        print(f"  Memória Total: {mem_total} MB")
                        print(f"  Memória Livre: {mem_free} MB")
                        print(f"  Utilização: {utilization}%")
                        
                        # Calcular memória usada
                        mem_used = int(mem_total) - int(mem_free)
                        mem_used_gb = mem_used / 1024
                        print(f"  Memória Usada: {mem_used_gb:.1f} GB")
                        
                        if mem_used_gb > 8:
                            print("  ⚠️  ALTA UTILIZAÇÃO DE MEMÓRIA")
                        elif mem_used_gb > 4:
                            print("  ⚠️  MÉDIA UTILIZAÇÃO DE MEMÓRIA")
                        else:
                            print("  ✅ BAIXA UTILIZAÇÃO DE MEMÓRIA")
                        
                        return True
                        
        print("❌ RTX 3080 Ti não encontrada")
        return False
            
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def optimize_for_lhr():
    """Otimiza configurações para GPU LHR"""
    
    print("\n⚡ Otimizando para RTX 3080 Ti LHR...")
    print("=" * 50)
    
    # Configurações específicas para LHR
    lhr_optimizations = [
        "set CUDA_VISIBLE_DEVICES=0",
        "set CUDA_LAUNCH_BLOCKING=0",
        "set CUDA_CACHE_DISABLE=0", 
        "set CUDA_FORCE_PTX_JIT=0",
        "set CUDA_MEMORY_POOL_SIZE=0",
        "set CUDA_DEVICE_ORDER=PCI_BUS_ID"
    ]
    
    print("📋 Configurações LHR recomendadas:")
    for opt in lhr_optimizations:
        print(f"  {opt}")
    
    # Criar script otimizado para LHR
    lhr_script = """@echo off
echo ========================================
echo    Multi-Pool Miner - RTX 3080 Ti LHR
echo ========================================
echo.

echo Configurando para RTX 3080 Ti LHR...
set CUDA_VISIBLE_DEVICES=0
set CUDA_LAUNCH_BLOCKING=0
set CUDA_CACHE_DISABLE=0
set CUDA_FORCE_PTX_JIT=0
set CUDA_MEMORY_POOL_SIZE=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID

echo.
echo IMPORTANTE: GPU LHR tem limitações de hashrate!
echo Performance esperada: 1.5-2.0 GH/s
echo.
pause

echo.
echo Iniciando minerador LHR otimizado...
echo.

python auto_miner.py

echo.
echo Minerador parado.
pause
"""
    
    with open("start_miner_lhr.bat", "w") as f:
        f.write(lhr_script)
    
    print("✅ Script LHR criado: start_miner_lhr.bat")

def test_lhr_performance():
    """Testa performance específica da LHR"""
    
    print("\n🔧 Testando performance LHR...")
    print("=" * 50)
    
    # Aplicar configurações LHR
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['CUDA_FORCE_PTX_JIT'] = '0'
    os.environ['CUDA_MEMORY_POOL_SIZE'] = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
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
        
        print(f"✅ Hashrate LHR: {hashrate_gh:.2f} GH/s")
        print(f"⏱️  Tempo: {duration:.2f} segundos")
        
        # Avaliar performance LHR
        if hashrate_gh > 2.0:
            print("🎉 Performance EXCELENTE para LHR!")
        elif hashrate_gh > 1.5:
            print("✅ Performance BOA para LHR")
        elif hashrate_gh > 1.0:
            print("⚠️ Performance ACEITÁVEL para LHR")
        else:
            print("❌ Performance BAIXA - verifique configurações")
            
        print(f"\n📊 Comparação LHR vs Normal:")
        print(f"  LHR Esperado: 1.5-2.0 GH/s")
        print(f"  Normal Esperado: 2.5-3.0 GH/s")
        print(f"  Seu Resultado: {hashrate_gh:.2f} GH/s")
        
    else:
        print("❌ Erro no teste LHR")

def create_lhr_monitor():
    """Cria monitor específico para LHR"""
    
    print("\n📊 Criando monitor LHR...")
    print("=" * 50)
    
    monitor_script = """#!/usr/bin/env python3
\"\"\"
Monitor específico para RTX 3080 Ti LHR
\"\"\"

import time
import json
import os
from datetime import datetime

def monitor_lhr():
    \"\"\"Monitora performance da LHR\"\"\"
    
    print("🎯 Monitor RTX 3080 Ti LHR")
    print("=" * 50)
    
    # Configurar para LHR
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    total_shares = 0
    start_time = datetime.now()
    
    while True:
        try:
            # Verificar miner.log
            if os.path.exists("miner.log"):
                with open("miner.log", "r") as f:
                    lines = f.readlines()
                    
                # Procurar por shares encontrados
                for line in lines:
                    if "Found valid nonce" in line:
                        total_shares += 1
                        print(f"🎯 SHARE ENCONTRADO! Total: {total_shares}")
                        print(f"   Tempo: {datetime.now()}")
                        print(f"   Runtime: {datetime.now() - start_time}")
                        print()
            
            # Mostrar status
            runtime = datetime.now() - start_time
            print(f"⏱️  Runtime: {runtime}")
            print(f"📈 Shares encontrados: {total_shares}")
            print(f"🎯 Hashrate esperado: 1.5-2.0 GH/s")
            print(f"💡 GPU: RTX 3080 Ti LHR")
            print("-" * 30)
            
            time.sleep(30)  # Atualizar a cada 30s
            
        except KeyboardInterrupt:
            print("\\n✅ Monitor parado")
            break
        except Exception as e:
            print(f"❌ Erro: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_lhr()
"""
    
    with open("monitor_lhr.py", "w") as f:
        f.write(monitor_script)
    
    print("✅ Monitor LHR criado: monitor_lhr.py")

if __name__ == "__main__":
    print("🎯 Otimizador RTX 3080 Ti LHR")
    print("=" * 50)
    
    # Verificar status LHR
    if check_lhr_status():
        # Otimizar para LHR
        optimize_for_lhr()
        
        # Testar performance LHR
        test_lhr_performance()
        
        # Criar monitor LHR
        create_lhr_monitor()
        
        print("\n✅ Otimização LHR concluída!")
        print("💡 Scripts disponíveis:")
        print("  - start_miner_lhr.bat (LHR otimizado)")
        print("  - monitor_lhr.py (monitor específico)")
        print("  - start_miner_optimized.bat (geral)")
        
        print("\n📋 Informações LHR:")
        print("  - Hashrate limitado: ~1.5-2.0 GH/s")
        print("  - Memória: 12GB GDDR6X")
        print("  - Limitações: Algoritmos específicos")
        print("  - Vantagem: Menor consumo de energia")
    else:
        print("❌ RTX 3080 Ti LHR não detectada") 