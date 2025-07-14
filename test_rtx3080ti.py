#!/usr/bin/env python3
"""
Teste de performance da RTX 3080 Ti
"""

import subprocess
import time
import os

def test_rtx3080ti_performance():
    """Testa performance da RTX 3080 Ti"""
    
    print("🚀 Teste de Performance - RTX 3080 Ti")
    print("=" * 50)
    
    # Configurar para usar apenas RTX 3080 Ti
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print("📋 Configuração:")
    print(f"  GPU: RTX 3080 Ti (CUDA_VISIBLE_DEVICES=0)")
    print(f"  Memória disponível: ~7GB")
    print()
    
    # Testar com targets fáceis primeiro
    print("🔧 Teste 1: Targets fáceis...")
    result = subprocess.run(
        ["python", "test_miner.py"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        print("✅ Teste com targets fáceis: OK")
        if "SUCCESS" in result.stdout:
            print("🎯 Encontrou shares válidos!")
    else:
        print("❌ Teste com targets fáceis: FALHOU")
    
    # Testar hashrate
    print("\n⚡ Teste 2: Medição de hashrate...")
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
        # Calcular hashrate estimado
        nonces_tested = 262144 * 1024  # 1024 blocks * 256 threads * 1024 nonces
        hashrate = nonces_tested / duration
        hashrate_gh = hashrate / 1_000_000_000
        
        print(f"✅ Hashrate estimado: {hashrate_gh:.2f} GH/s")
        print(f"⏱️  Tempo de execução: {duration:.2f} segundos")
        
        if hashrate_gh > 1.0:
            print("🎉 Performance excelente!")
        elif hashrate_gh > 0.5:
            print("✅ Performance boa")
        else:
            print("⚠️ Performance baixa - verifique conflitos de GPU")
    else:
        print("❌ Erro no teste de hashrate")
    
    # Testar com jobs reais
    print("\n🔧 Teste 3: Jobs reais das pools...")
    
    # Primeiro pegar jobs
    result = subprocess.run(
        ["python", "get_jobs.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        print("✅ Jobs obtidos com sucesso")
        
        # Testar minerador com jobs reais
        result = subprocess.run(
            ["./miner.exe", "0"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            if "Valid nonce found" in result.stdout:
                print("🎯 ENCONTROU SHARE VÁLIDO!")
                print("✅ Minerador funcionando perfeitamente!")
            else:
                print("✅ Minerador funcionando (sem shares neste teste)")
        else:
            print("❌ Erro no minerador")
    else:
        print("❌ Erro ao obter jobs")

def compare_gpus():
    """Compara performance das duas GPUs"""
    
    print("\n📊 Comparação de GPUs:")
    print("=" * 50)
    
    for gpu_id in [0, 1]:
        print(f"\n🔧 Testando GPU {gpu_id}...")
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        start_time = time.time()
        result = subprocess.run(
            ["./miner.exe", "0"],
            capture_output=True,
            text=True,
            timeout=10
        )
        end_time = time.time()
        
        if result.returncode == 0:
            duration = end_time - start_time
            nonces_tested = 262144 * 1024
            hashrate = nonces_tested / duration
            hashrate_gh = hashrate / 1_000_000_000
            
            gpu_name = "RTX 3080 Ti" if gpu_id == 0 else "GTX 1060"
            print(f"  {gpu_name}: {hashrate_gh:.2f} GH/s")
        else:
            print(f"  GPU {gpu_id}: Erro")

if __name__ == "__main__":
    print("🎯 Teste Específico - RTX 3080 Ti")
    print("=" * 50)
    
    # Testar performance da RTX 3080 Ti
    test_rtx3080ti_performance()
    
    # Comparar GPUs (opcional)
    print("\n" + "=" * 50)
    response = input("Deseja comparar performance das duas GPUs? (s/n): ")
    if response.lower() == 's':
        compare_gpus()
    
    print("\n✅ Teste concluído!")
    print("💡 Para minerar: execute start_miner_rtx3080ti.bat") 