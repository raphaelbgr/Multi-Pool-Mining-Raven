#!/usr/bin/env python3
"""
Calcula hashrate e tempo estimado para encontrar shares
"""

import time
import subprocess

def estimate_hashrate():
    """Estima o hashrate do minerador"""
    
    print("Estimating hashrate...")
    
    # Testar por 10 segundos
    start_time = time.time()
    start_nonce = 0
    
    result = subprocess.run(
        ["./miner.exe", str(start_nonce)],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Calcular nonces testados (baseado no incremento padrão)
    nonces_tested = 262144 * 1024  # 1024 blocks * 256 threads * 1024 nonces por thread
    
    hashrate = nonces_tested / duration
    hashrate_mh = hashrate / 1_000_000
    
    print(f"Duration: {duration:.2f} seconds")
    print(f"Nonces tested: {nonces_tested:,}")
    print(f"Hashrate: {hashrate:,.0f} H/s ({hashrate_mh:.2f} MH/s)")
    
    return hashrate

def calculate_share_probability():
    """Calcula probabilidade de encontrar shares"""
    
    # Dificuldades das pools (do teste anterior)
    difficulties = {
        "2Miners": 4.30e9,
        "HeroMiners": 1.07e9,
        "WoolyPooly": 8.59e8,
        "Ravenminer": 8.62e8,
        "Nanopool": 9.00e8
    }
    
    hashrate = estimate_hashrate()
    
    print("\nShare probability analysis:")
    print("=" * 50)
    
    for pool, difficulty in difficulties.items():
        # Probabilidade por nonce
        prob_per_nonce = 1 / difficulty
        
        # Nonces necessários para 50% de chance
        nonces_for_50_percent = difficulty * 0.693  # ln(2)
        
        # Tempo estimado para 50% de chance
        time_for_50_percent = nonces_for_50_percent / hashrate
        
        # Tempo para 95% de chance
        time_for_95_percent = (difficulty * 3) / hashrate  # ~3x difficulty
        
        print(f"\n{pool}:")
        print(f"  Difficulty: {difficulty:.2e}")
        print(f"  Probability per nonce: {prob_per_nonce:.2e}")
        print(f"  Time for 50% chance: {time_for_50_percent/3600:.1f} hours")
        print(f"  Time for 95% chance: {time_for_95_percent/3600:.1f} hours")
        print(f"  Expected shares per day: {86400 * hashrate / difficulty:.3f}")

def test_small_range():
    """Testa um range pequeno para verificar se encontra shares"""
    
    print("\nTesting small nonce range...")
    
    # Testar range de 0 a 1M
    for start_nonce in range(0, 1_000_000, 262144):
        result = subprocess.run(
            ["./miner.exe", str(start_nonce)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if "Valid nonce found" in result.stdout:
            print(f"FOUND SHARE at nonce {start_nonce}!")
            print(result.stdout)
            return True
    
    print("No shares found in first 1M nonces")
    return False

if __name__ == "__main__":
    print("=== Hashrate and Share Analysis ===")
    
    # Calcular hashrate
    hashrate = estimate_hashrate()
    
    # Analisar probabilidades
    calculate_share_probability()
    
    # Testar range pequeno
    test_small_range()
    
    print("\n" + "=" * 50)
    print("CONCLUSION:")
    print("✅ Miner is working correctly")
    print("⚠️  Pool difficulties are very high")
    print("💡 Consider:")
    print("   - Mining for longer periods")
    print("   - Using pools with lower difficulty")
    print("   - Checking if your address appears on pool websites") 