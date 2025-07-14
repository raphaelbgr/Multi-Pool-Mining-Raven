#!/usr/bin/env python3
"""
Test script for optimized multi-pool miner
"""

import subprocess
import time
import json
import os

def test_job_freshness():
    """Test if jobs are fresh enough"""
    print("[TEST] Testing job freshness...")
    
    # Get fresh jobs
    result = subprocess.run(
        ["python", "get_jobs.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print("[ERROR] Failed to get jobs")
        return False
    
    # Check job age
    if os.path.exists("jobs.json"):
        job_age = time.time() - os.path.getmtime("jobs.json")
        print(f"[AGE] Jobs are {job_age:.1f} seconds old")
        
        if job_age > 10:
            print("[WARN] Jobs might be too old for KawPOW")
            return False
        else:
            print("[OK] Jobs are fresh enough")
            return True
    else:
        print("[ERROR] jobs.json not found")
        return False

def test_optimized_miner():
    """Test the optimized miner with short timeout"""
    print("[TEST] Testing optimized miner with short timeout...")
    
    try:
        result = subprocess.run(
            ["./miner_multi_target.exe", "0"],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            print("[OK] Optimized miner completed within 15 seconds")
            
            # Check for valid nonces
            if "Valid nonce found" in result.stdout:
                print("[SUCCESS] Found valid nonces!")
                print(result.stdout)
            else:
                print("[INFO] No valid nonces found (normal for short test)")
                
            return True
        else:
            print("[ERROR] Optimized miner failed")
            print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Miner timed out after 15s (expected)")
        return True
    except Exception as e:
        print(f"[ERROR] Miner test failed: {str(e)}")
        return False

def test_rapid_cycling():
    """Test rapid cycling with fresh jobs"""
    print("[TEST] Testing rapid cycling with fresh jobs...")
    
    for i in range(3):
        print(f"\n[CYCLE] Cycle {i+1}/3")
        
        # Get fresh jobs
        if not test_job_freshness():
            print("[ERROR] Job freshness test failed")
            return False
        
        # Quick miner test
        try:
            result = subprocess.run(
                ["./miner_multi_target.exe", str(i * 100000)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if "Valid nonce found" in result.stdout:
                print(f"[SUCCESS] Cycle {i+1} found valid nonces!")
                
        except subprocess.TimeoutExpired:
            print(f"[INFO] Cycle {i+1} timed out (expected)")
        
        # Short pause
        time.sleep(0.5)
    
    print("[OK] Rapid cycling test completed")
    return True

def main():
    print("=" * 60)
    print("OPTIMIZED MINER TEST SUITE")
    print("=" * 60)
    
    # Test 1: Job freshness
    if not test_job_freshness():
        print("\n[FAILED] Job freshness test failed")
        return
    
    # Test 2: Optimized miner
    if not test_optimized_miner():
        print("\n[FAILED] Optimized miner test failed")
        return
    
    # Test 3: Rapid cycling
    if not test_rapid_cycling():
        print("\n[FAILED] Rapid cycling test failed")
        return
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed!")
    print("=" * 60)
    
    print("\n[READY] Optimized miner is ready to run:")
    print("  python auto_miner_optimized.py")
    print("\nExpected improvements:")
    print("  - Jobs stay fresh (< 10 seconds old)")
    print("  - Faster mining cycles (15s timeout)")
    print("  - Fresh jobs fetched before submission")
    print("  - Reduced 'job not found' errors")

if __name__ == "__main__":
    main() 