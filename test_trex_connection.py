#!/usr/bin/env python3
"""
Test script to simulate T-Rex miner connection and share submission
"""

import socket
import json
import time
import threading

def test_trex_connection():
    """Test if T-Rex can connect to the proxy"""
    print("[TEST] Testing T-Rex connection to proxy...")
    
    try:
        # Connect to proxy
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 4444))
        print("[OK] Connected to proxy on port 4444")
        
        # Send T-Rex authorization
        auth_message = {
            "id": 1,
            "method": "mining.authorize",
            "params": ["RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU", "x"]
        }
        sock.send((json.dumps(auth_message) + "\n").encode())
        
        # Read response
        response = sock.recv(1024).decode()
        print(f"[RESPONSE] Authorization response: {response.strip()}")
        
        # Wait for job
        print("[INFO] Waiting for job...")
        job_data = sock.recv(4096).decode()
        print(f"[JOB] Received job: {job_data[:200]}...")
        
        # Parse job
        try:
            job_lines = job_data.strip().split('\n')
            for line in job_lines:
                if line.strip():
                    job_msg = json.loads(line)
                    if job_msg.get('method') == 'mining.notify':
                        job_id = job_msg['params'][0]
                        print(f"[JOB] Job ID: {job_id}")
                        
                        # Simulate share submission
                        share_message = {
                            "id": 2,
                            "method": "mining.submit",
                            "params": [
                                "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU",  # worker
                                job_id,  # job_id
                                "0xae00000012345678",  # extranonce2
                                "abcdef0123456789" * 8,  # mixhash (64 chars)
                                "0x" + "12345678" * 16  # result (64 chars)
                            ]
                        }
                        
                        print(f"[SUBMIT] Submitting test share...")
                        sock.send((json.dumps(share_message) + "\n").encode())
                        
                        # Wait for submission response
                        time.sleep(2)
                        try:
                            share_response = sock.recv(1024).decode()
                            print(f"[SHARE] Share response: {share_response.strip()}")
                        except:
                            print("[INFO] No immediate share response (may be normal)")
                        
                        break
        except Exception as e:
            print(f"[ERROR] Failed to parse job: {e}")
        
        sock.close()
        print("[OK] T-Rex connection test completed")
        
    except Exception as e:
        print(f"[ERROR] T-Rex connection test failed: {e}")

def test_pool_reporting():
    """Check if pools are receiving and reporting shares"""
    print("\n[TEST] Testing pool reporting...")
    
    # Give some time for the proxy to process
    time.sleep(5)
    
    print("[INFO] Check your pool dashboards:")
    print("  - 2Miners: https://rvn.2miners.com/account/RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU")
    print("  - Ravenminer: https://ravenminer.com/site/wallet_miners_results?wallet=RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU")
    print("  - WoolyPooly: https://woolypooly.com/en/coin/rvn")
    print("  - HeroMiners: https://ravencoin.herominers.com/")
    print("  - Nanopool: https://rvn.nanopool.org/account/RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU")

if __name__ == "__main__":
    print("=" * 60)
    print("T-REX MINER CONNECTION TEST")
    print("=" * 60)
    
    # Test connection
    test_trex_connection()
    
    # Test pool reporting
    test_pool_reporting()
    
    print("\n[INFO] Test completed!")
    print("[INFO] If T-Rex connects but pools show nothing, the issue may be:")
    print("  1. Share difficulty too high")
    print("  2. Nonce extraction method not working")
    print("  3. Pool-specific formatting issues")
    print("  4. T-Rex not actually mining (just connecting)") 