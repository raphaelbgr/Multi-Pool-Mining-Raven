#!/usr/bin/env python3

import json
import sys
sys.path.append('.')

from adapters.factory import AdapterFactory

def debug_submission():
    """Debug exactly what's being sent to 2Miners"""
    
    # Get fresh jobs
    import subprocess
    print("Getting fresh jobs...")
    result = subprocess.run(['python', 'get_jobs.py'], capture_output=True, text=True)
    
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
    
    print(f"2Miners Pool Config:")
    print(f"  Host: {miners_pool['host']}:{miners_pool['port']}")
    print(f"  User: {miners_pool['user']}")
    print(f"  Adapter: {miners_pool['adapter']}")
    print(f"  Extra: {miners_pool['extra']}")
    print()
    
    print(f"Job Details:")
    print(f"  Job ID: {miners_job['job_id']}")
    print(f"  Target: {miners_job['target']}")
    print(f"  Header: {miners_job['header_hash']}")
    print(f"  Ntime: {miners_job['ntime']}")
    print(f"  Pool Index: {miners_job['pool_index']}")
    print()
    
    # Test different nonce values
    test_nonces = [
        1,           # Very simple
        123456,      # Simple number
        1000000,     # Round number
        1853120,     # Original failing nonce
        0xdeadbeef,  # Common test value
        0x12345678,  # Sequential hex
    ]
    
    for test_nonce in test_nonces:
        print(f"Testing nonce: {test_nonce} (0x{test_nonce:08x})")
        
        try:
            # Create adapter
            adapter = AdapterFactory.create_adapter(miners_pool)
            
            # Connect
            adapter.connect()
            if not adapter.connected:
                print("  Could not connect")
                continue
            
            # Format and show submission
            submission = adapter._format_submission(test_nonce, miners_job)
            print(f"  Submission: {submission}")
            
            # Try the submission
            response = adapter.submit_share(test_nonce, miners_job)
            print(f"  Response: {response}")
            
            if isinstance(response, dict) and response.get('error'):
                error = response['error']
                if isinstance(error, list) and len(error) > 1:
                    error_msg = error[1]
                    if error_msg != 'Malformed nonce':
                        print(f"  !! Different error: {error_msg}")
                    else:
                        print(f"  XX Still malformed nonce")
                else:
                    print(f"  XX Error: {error}")
            else:
                print(f"  OK Success or different response!")
                break  # Found a working nonce!
            
            adapter.close()
            
        except Exception as e:
            print(f"  Exception: {e}")
        
        print()

if __name__ == "__main__":
    debug_submission() 