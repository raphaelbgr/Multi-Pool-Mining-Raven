#!/usr/bin/env python3
"""
Monitor específico para RTX 3080 Ti LHR
"""

import time
import json
import os
from datetime import datetime

def monitor_lhr():
    """Monitora performance da LHR"""
    
    print("Monitor RTX 3080 Ti LHR")
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
                        print(f"SHARE ENCONTRADO! Total: {total_shares}")
                        print(f"   Tempo: {datetime.now()}")
                        print(f"   Runtime: {datetime.now() - start_time}")
                        print()
            
            # Mostrar status
            runtime = datetime.now() - start_time
            print(f"Runtime: {runtime}")
            print(f"Shares encontrados: {total_shares}")
            print(f"Hashrate esperado: 1.5-2.0 GH/s")
            print(f"GPU: RTX 3080 Ti LHR")
            print("-" * 30)
            
            time.sleep(30)  # Atualizar a cada 30s
            
        except KeyboardInterrupt:
            print("\nMonitor parado")
            break
        except Exception as e:
            print(f"Erro: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_lhr()
