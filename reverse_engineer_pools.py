#!/usr/bin/env python3
"""
Reverse Engineer Pool Protocols
Test each pool systematically to understand exact format requirements
"""

import socket
import json
import time
import threading
from itertools import product

def test_pool_handshake(pool_config):
    """Test pool handshake and capture exact response format"""
    print(f"\n=== REVERSE ENGINEERING {pool_config['name']} ===")
    
    try:
        sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
        print(f"[OK] Connected to {pool_config['name']}")
        
        # Test subscribe
        subscribe_msg = json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }) + "\n"
        
        sock.send(subscribe_msg.encode())
        print(f"[SEND] Subscribe: {subscribe_msg.strip()}")
        
        # Get subscribe response
        response = sock.recv(4096).decode()
        print(f"[RECV] Subscribe response:")
        for line in response.split('\n'):
            if line.strip():
                try:
                    parsed = json.loads(line)
                    print(f"  {json.dumps(parsed, indent=2)}")
                    
                    # Extract extra_nonce if available
                    if parsed.get('id') == 1 and 'result' in parsed:
                        result = parsed['result']
                        if isinstance(result, list) and len(result) >= 2:
                            extra_nonce = result[1]
                            print(f"[LEARN] Extra nonce: '{extra_nonce}' (length: {len(extra_nonce)})")
                except json.JSONDecodeError:
                    print(f"  Non-JSON: {line}")
        
        # Test authorization
        auth_msg = json.dumps({
            "id": 2,
            "method": "mining.authorize",
            "params": [pool_config['user'], pool_config['password']]
        }) + "\n"
        
        sock.send(auth_msg.encode())
        print(f"[SEND] Auth: {auth_msg.strip()}")
        
        # Get auth response and any additional messages
        time.sleep(2)  # Give pool time to send initial data
        auth_response = sock.recv(4096).decode()
        print(f"[RECV] Auth response:")
        
        job_data = None
        for line in auth_response.split('\n'):
            if line.strip():
                try:
                    parsed = json.loads(line)
                    print(f"  {json.dumps(parsed, indent=2)}")
                    
                    # Look for mining.notify (job)
                    if parsed.get('method') == 'mining.notify':
                        job_data = parsed
                        print(f"[LEARN] Got job notification!")
                        
                except json.JSONDecodeError:
                    print(f"  Non-JSON: {line}")
        
        # Wait for job if not received yet
        if not job_data:
            print(f"[WAIT] Waiting for job from {pool_config['name']}...")
            sock.settimeout(15)
            try:
                job_response = sock.recv(4096).decode()
                print(f"[RECV] Job data:")
                for line in job_response.split('\n'):
                    if line.strip():
                        try:
                            parsed = json.loads(line)
                            print(f"  {json.dumps(parsed, indent=2)}")
                            if parsed.get('method') == 'mining.notify':
                                job_data = parsed
                        except json.JSONDecodeError:
                            print(f"  Non-JSON: {line}")
            except socket.timeout:
                print(f"[WARN] No job received from {pool_config['name']} within 15s")
        
        sock.close()
        return job_data
        
    except Exception as e:
        print(f"[ERROR] Failed to test {pool_config['name']}: {e}")
        return None

def test_submission_formats(pool_config, job_data, extra_nonce=""):
    """Test different submission formats to find what works"""
    print(f"\n=== TESTING SUBMISSION FORMATS FOR {pool_config['name']} ===")
    
    if not job_data:
        print(f"[SKIP] No job data for {pool_config['name']}")
        return
    
    job_params = job_data['params']
    job_id = job_params[0]
    
    print(f"[JOB] Job ID: {job_id}")
    print(f"[JOB] Params count: {len(job_params)}")
    print(f"[JOB] Params: {job_params}")
    
    # Test nonce: 0x12345678 = 305419896 decimal
    test_nonce_decimal = 305419896
    
    # Different nonce formats to test
    nonce_formats = {
        "big_endian_8": f"{test_nonce_decimal:08x}",           # 12345678
        "little_endian_8": f"{(test_nonce_decimal & 0xFF):02x}{(test_nonce_decimal >> 8 & 0xFF):02x}{(test_nonce_decimal >> 16 & 0xFF):02x}{(test_nonce_decimal >> 24 & 0xFF):02x}",  # 78563412
        "big_endian_4": f"{test_nonce_decimal:04x}",           # 2345678 (truncated)
        "decimal_str": str(test_nonce_decimal),                 # "305419896"
        "hex_prefix": f"0x{test_nonce_decimal:08x}",           # 0x12345678
    }
    
    # Different extranonce2 formats to test
    extranonce2_formats = {
        "simple_4byte": "00000001",
        "simple_2byte": "0001", 
        "incremental": "12345678",
        "with_extra": extra_nonce + "00000001" if extra_nonce else "00000001",
        "long_8byte": "0000000000000001",
        "timestamp": f"{int(time.time()) & 0xFFFFFFFF:08x}",
    }
    
    # Different ntime formats
    current_time = int(time.time())
    ntime_formats = {
        "current_hex": f"{current_time:08x}",
        "job_ntime": job_params[6] if len(job_params) > 6 else job_params[5] if len(job_params) > 5 else f"{current_time:08x}",
        "current_decimal": str(current_time),
        "zero_padded": "00000000",
    }
    
    print(f"[TEST] Testing {len(nonce_formats)} nonce formats x {len(extranonce2_formats)} extranonce2 formats x {len(ntime_formats)} ntime formats")
    print(f"[TEST] Total combinations: {len(nonce_formats) * len(extranonce2_formats) * len(ntime_formats)}")
    
    # Test different combinations
    working_formats = []
    
    for (nonce_name, nonce_val), (ext2_name, ext2_val), (ntime_name, ntime_val) in product(
        nonce_formats.items(), extranonce2_formats.items(), ntime_formats.items()
    ):
        
        # Try this combination
        try:
            sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=5)
            
            # Quick handshake
            sock.send(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
            sock.recv(2048)  # Ignore response
            
            sock.send(json.dumps({"id": 2, "method": "mining.authorize", "params": [pool_config['user'], pool_config['password']]}).encode() + b"\n")
            sock.recv(2048)  # Ignore response
            
            # Submit test share
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    pool_config['user'],
                    job_id,
                    ext2_val,
                    ntime_val,
                    nonce_val
                ]
            }
            
            sock.send((json.dumps(submission) + "\n").encode())
            
            # Get response
            response = sock.recv(2048).decode()
            sock.close()
            
            # Parse response
            for line in response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 3:  # Our submission response
                            error = parsed.get('error')
                            if not error:
                                # Success!
                                working_format = {
                                    'nonce_format': nonce_name,
                                    'nonce_value': nonce_val,
                                    'extranonce2_format': ext2_name,
                                    'extranonce2_value': ext2_val,
                                    'ntime_format': ntime_name,
                                    'ntime_value': ntime_val,
                                    'response': parsed
                                }
                                working_formats.append(working_format)
                                print(f"[SUCCESS] {nonce_name} + {ext2_name} + {ntime_name} = {parsed}")
                            else:
                                error_msg = error[1] if isinstance(error, list) and len(error) > 1 else str(error)
                                if error_msg not in ['Malformed nonce', 'Stale share', 'Job not found', 'Duplicate share']:
                                    print(f"[DIFFERENT] {nonce_name} + {ext2_name} + {ntime_name} = {error_msg}")
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            continue  # Skip failed connections
        
        # Don't spam the pool
        time.sleep(0.1)
    
    print(f"\n[RESULTS] Found {len(working_formats)} working format(s) for {pool_config['name']}:")
    for i, fmt in enumerate(working_formats):
        print(f"  {i+1}. Nonce: {fmt['nonce_format']} ({fmt['nonce_value']})")
        print(f"     Extranonce2: {fmt['extranonce2_format']} ({fmt['extranonce2_value']})")
        print(f"     Ntime: {fmt['ntime_format']} ({fmt['ntime_value']})")
        print(f"     Response: {fmt['response']}")
    
    return working_formats

def reverse_engineer_all_pools():
    """Reverse engineer all pools"""
    
    with open("config.json") as f:
        config = json.load(f)
    
    results = {}
    
    for pool_config in config['pools']:
        print(f"\n{'='*60}")
        print(f"REVERSE ENGINEERING {pool_config['name'].upper()}")
        print(f"{'='*60}")
        
        # Test handshake and get job
        job_data = test_pool_handshake(pool_config)
        
        # Test submission formats
        working_formats = test_submission_formats(pool_config, job_data)
        
        results[pool_config['name']] = {
            'job_data': job_data,
            'working_formats': working_formats
        }
        
        print(f"\n[SUMMARY] {pool_config['name']}:")
        if working_formats:
            print(f"  [OK] Found {len(working_formats)} working formats")
        else:
            print(f"  [FAIL] No working formats found")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for pool_name, data in results.items():
        print(f"\n{pool_name}:")
        if data['working_formats']:
            best_format = data['working_formats'][0]
            print(f"  [RECOMMENDED] Nonce: {best_format['nonce_format']}")
            print(f"  [RECOMMENDED] Extranonce2: {best_format['extranonce2_format']}")
            print(f"  [RECOMMENDED] Ntime: {best_format['ntime_format']}")
        else:
            print(f"  [FAILED] Could not find working format")
    
    return results

if __name__ == "__main__":
    print("POOL PROTOCOL REVERSE ENGINEERING")
    print("This will systematically test each pool to find working formats")
    print("WARNING: This will make many test connections to pools")
    print()
    
    input("Press Enter to continue...")
    
    results = reverse_engineer_all_pools()
    
    print(f"\n[COMPLETE] Reverse engineering complete!")
    print(f"[NEXT] Use these results to fix the pool adapters") 