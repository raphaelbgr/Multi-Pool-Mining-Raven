#!/usr/bin/env python3
"""
Diagnostic script to check proxy status and pool connections
"""

import socket
import json
import time
import threading

def check_proxy_running():
    """Check if proxy is running on port 4444"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 4444))
        sock.close()
        return result == 0
    except:
        return False

def check_pool_connectivity():
    """Check if pools are reachable"""
    pools = [
        ("2Miners", "rvn.2miners.com", 6060),
        ("Ravenminer", "stratum.ravenminer.com", 3838),
        ("WoolyPooly", "pool.br.woolypooly.com", 55555),
        ("HeroMiners", "br.ravencoin.herominers.com", 1140),
        ("Nanopool", "rvn-us-east1.nanopool.org", 10400)
    ]
    
    print("[INFO] Checking pool connectivity...")
    for name, host, port in pools:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            status = "[OK]" if result == 0 else "[FAIL]"
            print(f"  {status} {name}: {host}:{port}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

def test_proxy_connection():
    """Test connecting to proxy like T-Rex would"""
    print("\n[INFO] Testing proxy connection...")
    
    if not check_proxy_running():
        print("[ERROR] Proxy is not running on port 4444!")
        print("[INFO] Start the proxy first: python multi_pool_proxy.py")
        return False
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(('127.0.0.1', 4444))
        print("[OK] Connected to proxy")
        
        # Send authorization
        auth_msg = {
            "id": 1,
            "method": "mining.authorize", 
            "params": ["RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU", "x"]
        }
        sock.send((json.dumps(auth_msg) + "\n").encode())
        
        # Check response
        response = sock.recv(1024).decode()
        print(f"[RESPONSE] {response.strip()}")
        
        # Wait for job with timeout
        print("[INFO] Waiting for job (10 seconds)...")
        sock.settimeout(10)
        try:
            job_data = sock.recv(4096).decode()
            if job_data:
                print(f"[JOB] Received data: {job_data[:100]}...")
                return True
            else:
                print("[WARN] No job data received")
                return False
        except socket.timeout:
            print("[WARN] Timeout waiting for job")
            return False
        
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return False
    finally:
        try:
            sock.close()
        except:
            pass

def check_log_files():
    """Check for log files that might give clues"""
    import os
    
    log_files = ['miner.log', 'get_jobs.log', 'submit_share.log']
    
    print("\n[INFO] Checking log files...")
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"[LOG] {log_file} ({len(lines)} lines):")
                        # Show last 3 lines
                        for line in lines[-3:]:
                            print(f"  {line.strip()}")
                    else:
                        print(f"[LOG] {log_file} is empty")
            except Exception as e:
                print(f"[ERROR] Reading {log_file}: {e}")
        else:
            print(f"[LOG] {log_file} not found")

if __name__ == "__main__":
    print("=" * 60)
    print("PROXY STATUS DIAGNOSTIC")
    print("=" * 60)
    
    # Check if proxy is running
    if check_proxy_running():
        print("[OK] Proxy is running on port 4444")
    else:
        print("[ERROR] Proxy is NOT running on port 4444")
        print("[INFO] Start with: python multi_pool_proxy.py")
    
    # Check pool connectivity
    check_pool_connectivity()
    
    # Test proxy connection
    test_proxy_connection()
    
    # Check log files
    check_log_files()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    print("\n[INFO] Common issues and solutions:")
    print("1. If proxy not running: Start with 'python multi_pool_proxy.py'")
    print("2. If pools unreachable: Check internet connection")
    print("3. If no jobs received: Pools might not be sending jobs")
    print("4. If connection drops: Check proxy logs for errors")
    print("5. If shares not appearing: Check nonce extraction methods") 