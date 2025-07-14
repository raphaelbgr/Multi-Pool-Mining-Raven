#!/usr/bin/env python3
"""
Script para verificar GPUs disponíveis e configurar para RTX 3080 Ti
"""

import subprocess
import json

def check_gpus():
    """Verifica GPUs disponíveis usando nvidia-smi"""
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("🔍 GPUs Disponíveis:")
            print("=" * 60)
            
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_id = parts[0]
                    name = parts[1]
                    mem_total = parts[2]
                    mem_free = parts[3]
                    utilization = parts[4]
                    
                    print(f"GPU {gpu_id}: {name}")
                    print(f"  Memória: {mem_free}/{mem_total} MB")
                    print(f"  Utilização: {utilization}%")
                    print()
                    
            return True
        else:
            print("❌ Erro ao executar nvidia-smi")
            return False
            
    except FileNotFoundError:
        print("❌ nvidia-smi não encontrado. Verifique se os drivers NVIDIA estão instalados.")
        return False
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def test_gpu_performance():
    """Testa performance de cada GPU"""
    
    print("⚡ Testando Performance das GPUs:")
    print("=" * 60)
    
    # Testar cada GPU disponível
    for gpu_id in range(2):  # Assumindo 2 GPUs
        print(f"\n🔧 Testando GPU {gpu_id}...")
        
        try:
            # Definir variável de ambiente para GPU específica
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            # Testar minerador na GPU específica
            result = subprocess.run(
                ["./miner.exe", "0"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if result.returncode == 0:
                if "Valid nonce found" in result.stdout:
                    print(f"✅ GPU {gpu_id}: Funcionando (encontrou nonce)")
                else:
                    print(f"✅ GPU {gpu_id}: Funcionando (sem nonce válido)")
            else:
                print(f"❌ GPU {gpu_id}: Erro")
                
        except Exception as e:
            print(f"❌ GPU {gpu_id}: Erro - {str(e)}")

def configure_for_rtx3080ti():
    """Configura minerador para RTX 3080 Ti"""
    
    print("\n🎯 Configurando para RTX 3080 Ti:")
    print("=" * 60)
    
    # RTX 3080 Ti geralmente é GPU 0 (mais potente)
    target_gpu = 0
    
    print(f"📋 Configuração:")
    print(f"  GPU Alvo: {target_gpu} (RTX 3080 Ti)")
    print(f"  Variável: CUDA_VISIBLE_DEVICES={target_gpu}")
    
    # Criar script de inicialização
    startup_script = f"""@echo off
echo Iniciando minerador na RTX 3080 Ti...
set CUDA_VISIBLE_DEVICES={target_gpu}
python auto_miner.py
pause
"""
    
    with open("start_miner_rtx3080ti.bat", "w") as f:
        f.write(startup_script)
    
    print(f"✅ Script criado: start_miner_rtx3080ti.bat")
    print(f"💡 Execute: start_miner_rtx3080ti.bat")

if __name__ == "__main__":
    import os
    
    print("🚀 Configurador de GPU para Multi-Pool Miner")
    print("=" * 60)
    
    # Verificar GPUs
    if check_gpus():
        # Configurar para RTX 3080 Ti
        configure_for_rtx3080ti()
        
        print("\n📋 Próximos passos:")
        print("1. Execute: start_miner_rtx3080ti.bat")
        print("2. Ou manualmente: set CUDA_VISIBLE_DEVICES=0 && python auto_miner.py")
        print("3. Para testar: set CUDA_VISIBLE_DEVICES=0 && python test_miner.py")
    else:
        print("❌ Não foi possível verificar GPUs") 