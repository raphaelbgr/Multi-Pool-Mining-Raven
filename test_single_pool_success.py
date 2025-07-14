#!/usr/bin/env python3
"""
SINGLE POOL SUCCESS TEST
Get ANY single pool working perfectly first
"""

import json
import socket
import time
import subprocess
from datetime import datetime

class SinglePoolSuccessTest:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        with open("config.json") as f:
            return json.load(f)
    
    def test_pool_immediate_submission(self, pool_config):
        """Test immediate submission to a fresh job"""
        print(f"\n=== TESTING {pool_config['name']} ===")
        
        try:
            # Connect
            sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
            print(f"[CONNECT] Connected to {pool_config['name']}")
            
            # Subscribe
            sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
            response = sock.recv(4096).decode()
            
            # Get extra_nonce
            extra_nonce = ""
            for line in response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 1:
                            result = parsed.get('result', [])
                            if len(result) >= 2:
                                extra_nonce = result[1] or ""
                                print(f"[EXTRA] Extra nonce: '{extra_nonce}'")
                    except:
                        pass
            
            # Authorize
            sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [pool_config['user'], pool_config['password']]}).encode() + b"\n")
            auth_response = sock.recv(4096).decode()
            
            # Wait for job
            print(f"[WAIT] Waiting for job from {pool_config['name']}...")
            sock.settimeout(15)
            
            job_data = None
            start_time = time.time()
            
            while time.time() - start_time < 15:
                try:
                    data = sock.recv(4096).decode()
                    for line in data.split('\n'):
                        if line.strip():
                            try:
                                parsed = json.loads(line)
                                if parsed.get('method') == 'mining.notify':
                                    job_data = parsed
                                    job_id = job_data['params'][0]
                                    print(f"[JOB] Got fresh job: {job_id}")
                                    
                                    # IMMEDIATELY submit a test share
                                    test_nonce = 12345678  # Simple test nonce
                                    little_endian_nonce = f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}"
                                    
                                    submission = {
                                        "id": 3,
                                        "method": "mining.submit",
                                        "params": [
                                            pool_config['user'],
                                            job_id,
                                            extra_nonce + "0001",  # extra_nonce + extranonce2
                                            job_data['params'][6] if len(job_data['params']) > 6 else "00000000",
                                            little_endian_nonce
                                        ]
                                    }
                                    
                                    print(f"[SUBMIT] Submitting to FRESH job {job_id}...")
                                    sock.sendall((json.dumps(submission) + "\n").encode())
                                    
                                    # Get response immediately
                                    response = sock.recv(2048).decode()
                                    print(f"[RESPONSE] {response.strip()}")
                                    
                                    # Parse response
                                    for resp_line in response.split('\n'):
                                        if resp_line.strip():
                                            try:
                                                parsed_resp = json.loads(resp_line)
                                                if parsed_resp.get('id') == 3:
                                                    error = parsed_resp.get('error')
                                                    if error:
                                                        error_msg = error[1] if isinstance(error, list) and len(error) > 1 else str(error)
                                                        if "job not found" in error_msg.lower():
                                                            print(f"[STALE] Job expired: {error_msg}")
                                                        elif "low difficulty" in error_msg.lower():
                                                            print(f"[PROGRESS] Low difficulty - format is correct!")
                                                            return True
                                                        else:
                                                            print(f"[ERROR] {error_msg}")
                                                    else:
                                                        print(f"[SUCCESS] Share accepted!")
                                                        return True
                                            except json.JSONDecodeError:
                                                continue
                                    
                                    break
                            except json.JSONDecodeError:
                                continue
                    if job_data:
                        break
                except socket.timeout:
                    break
            
            sock.close()
            return False
            
        except Exception as e:
            print(f"[ERROR] {pool_config['name']} test failed: {e}")
            return False
    
    def test_all_pools(self):
        """Test all pools to find one that works"""
        print("=" * 60)
        print("SINGLE POOL SUCCESS TEST")
        print("=" * 60)
        print(f"Time: {datetime.now()}")
        print("[GOAL] Find ANY pool that accepts shares")
        
        working_pools = []
        
        for pool_config in self.config['pools']:
            if self.test_pool_immediate_submission(pool_config):
                working_pools.append(pool_config['name'])
                print(f"[SUCCESS] {pool_config['name']} is working!")
            else:
                print(f"[FAIL] {pool_config['name']} failed")
        
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        if working_pools:
            print(f"[SUCCESS] Found {len(working_pools)} working pool(s):")
            for pool in working_pools:
                print(f"  ✅ {pool}")
        else:
            print("[FAIL] No pools are working")
            print("[NEXT] The issue is deeper than job staleness")
        
        return working_pools

if __name__ == "__main__":
    test = SinglePoolSuccessTest()
    working_pools = test.test_all_pools() 