#!/usr/bin/env python3
"""
Monitor de shares encontrados e submetidos
"""

import time
import json
import os
from datetime import datetime

def monitor_shares():
    """Monitora shares encontrados em tempo real"""
    
    print("🔍 Share Monitor Started")
    print("=" * 50)
    
    # Estatísticas
    total_shares = 0
    accepted_shares = 0
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
                        print(f"🎯 SHARE FOUND: {line.strip()}")
                        
                    if "Share accepted by" in line:
                        accepted_shares += 1
                        print(f"✅ SHARE ACCEPTED: {line.strip()}")
                        
                    if "Share rejected by" in line:
                        print(f"❌ SHARE REJECTED: {line.strip()}")
            
            # Verificar submit_share.log
            if os.path.exists("submit_share.log"):
                with open("submit_share.log", "r") as f:
                    lines = f.readlines()
                    
                for line in lines:
                    if "Share accepted by" in line and "INFO" in line:
                        print(f"📤 SUBMISSION SUCCESS: {line.strip()}")
                        
            # Mostrar estatísticas
            runtime = datetime.now() - start_time
            print(f"\n📊 STATISTICS:")
            print(f"   Runtime: {runtime}")
            print(f"   Total shares found: {total_shares}")
            print(f"   Accepted shares: {accepted_shares}")
            print(f"   Acceptance rate: {(accepted_shares/total_shares*100):.1f}%" if total_shares > 0 else "   Acceptance rate: 0%")
            
            # Calcular hashrate estimado
            if total_shares > 0:
                avg_time_per_share = runtime.total_seconds() / total_shares
                print(f"   Average time per share: {avg_time_per_share:.1f} seconds")
            
            print("=" * 50)
            
            time.sleep(30)  # Verificar a cada 30 segundos
            
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            time.sleep(5)

def check_pool_status():
    """Verifica status das pools"""
    
    print("🔗 Checking pool connections...")
    
    try:
        result = subprocess.run(
            ["python", "get_jobs.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Pool connections working")
            
            # Verificar jobs.json
            if os.path.exists("jobs.json"):
                with open("jobs.json") as f:
                    jobs = json.load(f)
                    
                active_pools = len([j for j in jobs if j.get('header_hash')])
                print(f"📊 Active pools: {active_pools}/{len(jobs)}")
                
                for job in jobs:
                    if job.get('header_hash'):
                        print(f"   ✅ {job['pool_name']}: {job['job_id']}")
                    else:
                        print(f"   ❌ {job['pool_name']}: No job")
        else:
            print("❌ Pool connection issues")
            
    except Exception as e:
        print(f"❌ Error checking pools: {str(e)}")

if __name__ == "__main__":
    import subprocess
    
    print("🚀 Multi-Pool CUDA Miner Monitor")
    print("=" * 50)
    
    # Verificar status das pools
    check_pool_status()
    
    print("\n📈 Starting share monitor...")
    print("Press Ctrl+C to stop")
    
    monitor_shares() 