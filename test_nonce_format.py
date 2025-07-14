#!/usr/bin/env python3

import json
import sys
sys.path.append('.')

from adapters.factory import AdapterFactory

def test_nonce_variations():
    """Test different nonce formats and lengths with 2Miners"""
    
    # Get fresh jobs first
    import subprocess
    print("Getting fresh jobs...")
    result = subprocess.run(['python', 'get_jobs.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to get fresh jobs")
        return
    
    # Load current jobs
    with open("jobs.json") as f:
        jobs = json.load(f)
    
    # Load config
    with open("config.json") as f:
        config = json.load(f)
    
    # Find 2Miners pool and job
    miners_pool = None
    miners_job = None
    
    for pool in config['pools']:
        if pool['name'] == '2Miners':
            miners_pool = pool
            break
    
    for job in jobs:
        if job['pool_index'] == 0:  # 2Miners is pool 0
            miners_job = job
            break
    
    if not miners_pool or not miners_job:
        print("2Miners pool or job not found")
        return
    
    print(f"Testing with 2Miners")
    print(f"Job ID: {miners_job['job_id']}")
    print(f"Target: {miners_job['target'][:16]}...")
    print()
    
    # Test different nonce values and formats
    test_cases = [
        ("Small nonce", 123456),
        ("Original nonce", 1853120),
        ("Different nonce", 987654321),
        ("Very small", 1),
        ("16-bit max", 65535),
        ("24-bit max", 16777215),
    ]
    
    for case_name, test_nonce in test_cases:
        print(f"Testing {case_name}: {test_nonce}")
        
        # Test different formats for this nonce
        formats = {
            "Big-endian 8-char": f"{test_nonce:08x}",
            "Little-endian 8-char": f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}",
            "Big-endian 4-char": f"{test_nonce:04x}",
            "Uppercase": f"{test_nonce:08X}",
            "No padding": f"{test_nonce:x}",
        }
        
        for format_name, nonce_str in formats.items():
            print(f"  Format {format_name}: {nonce_str}")
            
            try:
                # Create adapter
                adapter = AdapterFactory.create_adapter(miners_pool)
                
                # Manually override the nonce formatting
                adapter._format_nonce = lambda n: nonce_str
                
                # Connect and test
                adapter.connect()
                if not adapter.connected:
                    print(f"    XX Could not connect")
                    continue
                
                # Submit test share
                response = adapter.submit_share(test_nonce, miners_job)
                
                if isinstance(response, dict) and response.get('error'):
                    error = response['error']
                    if isinstance(error, list) and len(error) > 1:
                        error_msg = error[1]
                        if error_msg != 'Malformed nonce':
                            print(f"    !! Different error: {error_msg}")
                        # Skip logging "Malformed nonce" to reduce noise
                    else:
                        print(f"    XX Error: {error}")
                else:
                    print(f"    OK Success or different response: {response}")
                
                adapter.close()
                
            except Exception as e:
                print(f"    XX Exception: {e}")
        
        print()

if __name__ == "__main__":
    test_nonce_variations() 