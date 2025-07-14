#!/usr/bin/env python3
"""
Script para testar o minerador com targets mais fáceis
"""

import json
import subprocess
import struct

def create_test_data():
    """Cria dados de teste com targets mais fáceis"""
    
    # Headers de teste (mesmo header para todos)
    test_header = "0000000000000000000000000000000000000000000000000000000000000000"
    
    # Targets mais fáceis (maior = mais fácil)
    easy_targets = [
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",  # Muito fácil
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff00",  # Fácil
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0000",  # Médio
    ]
    
    # Criar dados binários
    binary_data = bytearray()
    
    for i, target in enumerate(easy_targets):
        # Header length (32 bytes)
        binary_data.extend(struct.pack("<I", 32))
        # Header (32 bytes)
        binary_data.extend(bytes.fromhex(test_header))
        # Target (32 bytes)
        binary_data.extend(bytes.fromhex(target))
    
    # Salvar em headers.bin
    with open("headers.bin", "wb") as f:
        f.write(binary_data)
    
    # Criar jobs.json de teste
    test_jobs = []
    for i, target in enumerate(easy_targets):
        test_jobs.append({
            "pool_index": i,
            "pool_name": f"TestPool{i}",
            "job_id": f"test{i}",
            "header_hash": test_header,
            "target": target,
            "ntime": "1b010a76"
        })
    
    with open("jobs.json", "w") as f:
        json.dump(test_jobs, f, indent=2)
    
    print(f"Created test data with {len(easy_targets)} easy targets")
    print(f"Binary size: {len(binary_data)} bytes")

def test_miner():
    """Testa o minerador com targets fáceis"""
    
    print("Testing miner with easy targets...")
    
    # Testar com nonce 0
    result = subprocess.run(
        ["./miner.exe", "0"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    print("Miner output:")
    print(result.stdout)
    
    if "Valid nonce found" in result.stdout:
        print("SUCCESS: Found valid nonce!")
        return True
    else:
        print("No valid nonces found with easy targets")
        return False

def analyze_targets():
    """Analisa a dificuldade dos targets atuais"""
    
    with open("jobs.json") as f:
        jobs = json.load(f)
    
    print("Current target difficulties:")
    for job in jobs:
        target = job['target']
        # Converter para inteiro para comparar dificuldade
        target_int = int(target, 16)
        difficulty = 2**256 / (target_int + 1) if target_int > 0 else float('inf')
        
        print(f"  {job['pool_name']}: {target[:16]}... (difficulty: {difficulty:.2e})")

if __name__ == "__main__":
    print("=== Miner Test Script ===")
    
    # Analisar targets atuais
    print("\n1. Analyzing current targets:")
    analyze_targets()
    
    # Criar dados de teste
    print("\n2. Creating test data with easy targets:")
    create_test_data()
    
    # Testar minerador
    print("\n3. Testing miner:")
    success = test_miner()
    
    if success:
        print("\n✅ Miner is working correctly!")
    else:
        print("\n❌ Miner may have issues with hash calculation")
        print("This could indicate:")
        print("  - GPU driver issues")
        print("  - CUDA compilation problems")
        print("  - Memory allocation issues") 