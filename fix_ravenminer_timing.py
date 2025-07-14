#!/usr/bin/env python3
"""
FIX: Ravenminer Job Timing Issues
The problem is jobs are expiring before submission
"""

import json
import socket
import time
import subprocess
from datetime import datetime

class RavenminerTimingFix:
    def __init__(self):
        self.config = self.load_config()
        self.ravenminer_config = None
        
        for pool in self.config['pools']:
            if pool['name'] == 'Ravenminer':
                self.ravenminer_config = pool
                break
    
    def load_config(self):
        with open("config.json") as f:
            return json.load(f)
    
    def test_fresh_job_submission(self):
        """Test submitting to a FRESH job immediately after receiving it"""
        print("\n[FIX] Testing FRESH job submission...")
        
        try:
            sock = socket.create_connection(
                (self.ravenminer_config['host'], self.ravenminer_config['port']), 
                timeout=10
            )
            
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
                    except:
                        pass
            
            # Authorize
            sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [self.ravenminer_config['user'], self.ravenminer_config['password']]}).encode() + b"\n")
            sock.recv(4096)
            
            # Wait for job
            print("[WAIT] Waiting for fresh job...")
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
                                    print(f"[FRESH] Got job {job_id} at {datetime.now()}")
                                    
                                    # IMMEDIATELY submit share to this fresh job
                                    test_nonce = 305419896
                                    little_endian_nonce = f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}"
                                    
                                    submission = {
                                        "id": 3,
                                        "method": "mining.submit",
                                        "params": [
                                            self.ravenminer_config['user'],
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
            print(f"[ERROR] Fresh job test failed: {e}")
            return False
    
    def test_rapid_cycling(self):
        """Test rapid job cycling to keep jobs fresh"""
        print("\n[FIX] Testing rapid job cycling...")
        
        for cycle in range(3):
            print(f"\n[CYCLE] {cycle + 1}/3")
            
            # Get fresh jobs
            print("[FETCH] Getting fresh jobs...")
            result = subprocess.run(['python', 'get_jobs.py'], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"[ERROR] get_jobs.py failed: {result.stderr}")
                continue
            
            # Load jobs
            try:
                with open('jobs.json') as f:
                    jobs = json.load(f)
                
                # Find Ravenminer job
                ravenminer_job = None
                for job in jobs:
                    if job.get('pool_name') == 'Ravenminer':
                        ravenminer_job = job
                        break
                
                if not ravenminer_job:
                    print("[ERROR] No Ravenminer job found")
                    continue
                
                print(f"[JOB] Fresh job: {ravenminer_job['job_id']}")
                
                # Submit immediately
                from adapters.factory import AdapterFactory
                adapter = AdapterFactory.create_adapter(self.ravenminer_config)
                
                adapter.connect()
                if not adapter.connected:
                    print("[ERROR] Adapter connection failed")
                    continue
                
                # Test submission with fresh job
                test_nonce = 305419896
                print(f"[SUBMIT] Submitting to fresh job...")
                
                response = adapter.submit_share(test_nonce, ravenminer_job)
                print(f"[RESPONSE] {response}")
                
                adapter.close()
                
                # Check if successful
                if isinstance(response, dict):
                    if response.get('result') == True:
                        print("[SUCCESS] Share accepted!")
                        return True
                    elif response.get('error'):
                        error = response['error']
                        if isinstance(error, list) and len(error) > 1:
                            error_msg = error[1]
                            if "job not found" in error_msg.lower():
                                print(f"[STALE] Job expired: {error_msg}")
                            else:
                                print(f"[ERROR] {error_msg}")
                        else:
                            print(f"[ERROR] {error}")
                
            except Exception as e:
                print(f"[ERROR] Cycle {cycle + 1} failed: {e}")
            
            # Short pause between cycles
            time.sleep(2)
        
        return False
    
    def test_optimized_timing(self):
        """Test optimized timing with shorter mining cycles"""
        print("\n[FIX] Testing optimized timing...")
        
        # Modify auto_miner_optimized.py to use shorter cycles
        print("[MODIFY] Using 5-second mining cycles instead of 15s...")
        
        # Test with shorter timeout
        try:
            result = subprocess.run(
                ["./miner_multi_target.exe", "0"],
                capture_output=True,
                text=True,
                timeout=5  # Very short timeout
            )
            
            if result.returncode == 0:
                print("[OK] Short mining cycle completed")
                print("Output:")
                print(result.stdout)
                
                # Check if we found any nonces
                if "Valid nonce found" in result.stdout:
                    print("[SUCCESS] Found valid nonce in short cycle!")
                    return True
                else:
                    print("[INFO] No nonces found in short cycle (normal)")
            else:
                print(f"[ERROR] Short mining failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("[OK] Mining timed out as expected (5s)")
        except Exception as e:
            print(f"[ERROR] Short mining test failed: {e}")
        
        return False
    
    def run_all_fixes(self):
        """Run all timing fixes"""
        print("=" * 60)
        print("RAVENMINER TIMING FIXES")
        print("=" * 60)
        print(f"Time: {datetime.now()}")
        
        # Fix 1: Fresh job submission
        if self.test_fresh_job_submission():
            print("[SUCCESS] Fresh job submission works!")
            return True
        
        # Fix 2: Rapid cycling
        if self.test_rapid_cycling():
            print("[SUCCESS] Rapid cycling works!")
            return True
        
        # Fix 3: Optimized timing
        if self.test_optimized_timing():
            print("[SUCCESS] Optimized timing works!")
            return True
        
        print("\n[FAIL] All timing fixes failed")
        print("[NEXT] The issue may be deeper than timing")
        return False

if __name__ == "__main__":
    fix = RavenminerTimingFix()
    fix.run_all_fixes() 