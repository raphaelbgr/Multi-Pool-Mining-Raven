#!/usr/bin/env python3
"""
FOCUS TEST: Get Ravenminer working perfectly first
Test only Ravenminer pool to isolate and fix any issues
"""

import json
import socket
import time
import subprocess
from datetime import datetime

class RavenminerFocusTest:
    def __init__(self):
        self.config = self.load_config()
        self.ravenminer_config = None
        
        # Find Ravenminer config
        for pool in self.config['pools']:
            if pool['name'] == 'Ravenminer':
                self.ravenminer_config = pool
                break
        
        if not self.ravenminer_config:
            raise ValueError("Ravenminer not found in config")
        
        print(f"[FOCUS] Testing Ravenminer: {self.ravenminer_config['host']}:{self.ravenminer_config['port']}")
        print(f"[FOCUS] User: {self.ravenminer_config['user']}")
        print(f"[FOCUS] Worker name included: {'.' in self.ravenminer_config['user']}")
        
    def load_config(self):
        with open("config.json") as f:
            return json.load(f)
            
    def test_connection(self):
        """Test basic connection to Ravenminer"""
        print("\n[TEST] Testing connection to Ravenminer...")
        
        try:
            sock = socket.create_connection(
                (self.ravenminer_config['host'], self.ravenminer_config['port']), 
                timeout=10
            )
            print("[OK] Connected successfully")
            sock.close()
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False
            
    def test_handshake(self):
        """Test handshake process"""
        print("\n[TEST] Testing handshake...")
        
        try:
            sock = socket.create_connection(
                (self.ravenminer_config['host'], self.ravenminer_config['port']), 
                timeout=10
            )
            
            # Subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            sock.sendall((json.dumps(subscribe_msg) + "\n").encode())
            print("[SEND] Subscribe message sent")
            
            # Get subscribe response
            response = sock.recv(4096).decode()
            print(f"[RECV] Subscribe response: {response.strip()}")
            
            # Parse extra_nonce
            extra_nonce = ""
            for line in response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 1:
                            result = parsed.get('result', [])
                            if len(result) >= 2:
                                extra_nonce = result[1] or ""
                                print(f"[LEARN] Extra nonce: '{extra_nonce}' (length: {len(extra_nonce)})")
                            break
                    except json.JSONDecodeError:
                        continue
            
            # Authorize
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [self.ravenminer_config['user'], self.ravenminer_config['password']]
            }
            sock.sendall((json.dumps(auth_msg) + "\n").encode())
            print("[SEND] Authorization message sent")
            
            # Get auth response
            auth_response = sock.recv(4096).decode()
            print(f"[RECV] Auth response: {auth_response.strip()}")
            
            # Parse auth result
            auth_success = False
            for line in auth_response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 2:
                            auth_success = parsed.get('result', False)
                            print(f"[RESULT] Auth success: {auth_success}")
                            break
                    except json.JSONDecodeError:
                        continue
                        
            sock.close()
            return auth_success, extra_nonce
            
        except Exception as e:
            print(f"[ERROR] Handshake failed: {e}")
            return False, ""
            
    def test_job_fetching(self):
        """Test job fetching"""
        print("\n[TEST] Testing job fetching...")
        
        try:
            sock = socket.create_connection(
                (self.ravenminer_config['host'], self.ravenminer_config['port']), 
                timeout=10
            )
            
            # Quick handshake
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
            sock.recv(4096)  # consume auth response
            
            # Wait for job
            print("[WAIT] Waiting for job...")
            sock.settimeout(15)  # 15 second timeout for job
            
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
                                    print(f"[JOB] Received job: {job_data['params'][0]}")
                                    print(f"[JOB] Params: {len(job_data['params'])} parameters")
                                    break
                            except json.JSONDecodeError:
                                continue
                    if job_data:
                        break
                except socket.timeout:
                    break
                    
            sock.close()
            
            if job_data:
                print("[OK] Job received successfully")
                return job_data, extra_nonce
            else:
                print("[ERROR] No job received")
                return None, extra_nonce
                
        except Exception as e:
            print(f"[ERROR] Job fetching failed: {e}")
            return None, ""
            
    def test_share_submission(self, job_data, extra_nonce):
        """Test share submission with various nonce formats"""
        print("\n[TEST] Testing share submission...")
        
        if not job_data:
            print("[SKIP] No job data for testing")
            return
            
        job_params = job_data['params']
        job_id = job_params[0]
        print(f"[JOB] Job ID: {job_id}")
        
        # Test nonce: 305419896 (0x12345678)
        test_nonce = 305419896
        
        # Different nonce formats to test
        nonce_formats = {
            "little_endian": f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}",
            "big_endian": f"{test_nonce:08x}",
            "decimal": str(test_nonce),
            "hex_prefix": f"0x{test_nonce:08x}"
        }
        
        # Test extranonce2 formats
        extranonce2_formats = {
            "simple": "0001",
            "incremental": "1234", 
            "timestamp": f"{int(time.time()) & 0xFFFF:04x}",
            "zero_pad": "0000"
        }
        
        print(f"[TEST] Testing {len(nonce_formats)} nonce formats x {len(extranonce2_formats)} extranonce2 formats")
        
        for nonce_name, nonce_val in nonce_formats.items():
            for ext2_name, ext2_val in extranonce2_formats.items():
                print(f"\n[TRY] Nonce: {nonce_name} ({nonce_val}), Extranonce2: {ext2_name} ({ext2_val})")
                
                try:
                    sock = socket.create_connection(
                        (self.ravenminer_config['host'], self.ravenminer_config['port']), 
                        timeout=5
                    )
                    
                    # Quick handshake
                    sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
                    sock.recv(2048)
                    
                    sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [self.ravenminer_config['user'], self.ravenminer_config['password']]}).encode() + b"\n")
                    sock.recv(2048)
                    
                    # Submit share
                    submission = {
                        "id": 3,
                        "method": "mining.submit",
                        "params": [
                            self.ravenminer_config['user'],
                            job_id,
                            extra_nonce + ext2_val,  # Pool's extra_nonce + our extranonce2
                            job_params[6] if len(job_params) > 6 else "00000000",  # ntime
                            nonce_val
                        ]
                    }
                    
                    sock.sendall((json.dumps(submission) + "\n").encode())
                    
                    # Get response
                    response = sock.recv(2048).decode()
                    
                    # Parse response
                    for line in response.split('\n'):
                        if line.strip():
                            try:
                                parsed = json.loads(line)
                                if parsed.get('id') == 3:
                                    error = parsed.get('error')
                                    if error:
                                        error_msg = error[1] if isinstance(error, list) and len(error) > 1 else str(error)
                                        if error_msg == 'Job not found':
                                            print(f"    [STALE] Job expired - {error_msg}")
                                        else:
                                            print(f"    [ERROR] {error_msg}")
                                    else:
                                        print(f"    [SUCCESS] Share accepted! Response: {parsed}")
                                        return True  # Found working format!
                            except json.JSONDecodeError:
                                continue
                    
                    sock.close()
                    
                except Exception as e:
                    print(f"    [CONN_ERROR] {e}")
                
                # Don't spam the pool
                time.sleep(0.2)
        
        print("[RESULT] No working format found")
        return False
        
    def test_with_current_system(self):
        """Test with current get_jobs.py system"""
        print("\n[TEST] Testing with current get_jobs.py system...")
        
        # Get jobs using current system
        print("[FETCH] Running get_jobs.py...")
        result = subprocess.run(['python', 'get_jobs.py'], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] get_jobs.py failed: {result.stderr}")
            return False
            
        print("[OK] get_jobs.py completed")
        
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
                return False
                
            print(f"[JOB] Found Ravenminer job: {ravenminer_job['job_id']}")
            print(f"[JOB] Pool index: {ravenminer_job['pool_index']}")
            
            # Test submission using current adapter
            from adapters.factory import AdapterFactory
            
            adapter = AdapterFactory.create_adapter(self.ravenminer_config)
            print(f"[ADAPTER] Created adapter: {type(adapter).__name__}")
            
            # Connect
            adapter.connect()
            if not adapter.connected:
                print("[ERROR] Adapter connection failed")
                return False
                
            print("[OK] Adapter connected")
            
            # Test submission
            test_nonce = 305419896
            print(f"[SUBMIT] Testing nonce: {test_nonce}")
            
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
                        print(f"[ERROR] {error[1]}")
                    else:
                        print(f"[ERROR] {error}")
                else:
                    print(f"[UNKNOWN] Unknown response: {response}")
            else:
                print(f"[UNKNOWN] Unexpected response type: {type(response)}")
                
            return False
            
        except Exception as e:
            print(f"[ERROR] Current system test failed: {e}")
            return False
            
    def run_complete_test(self):
        """Run complete test sequence"""
        print("=" * 60)
        print("RAVENMINER FOCUS TEST")
        print("=" * 60)
        print(f"Time: {datetime.now()}")
        
        # Test 1: Basic connection
        if not self.test_connection():
            print("[FAIL] Basic connection test failed")
            return False
            
        # Test 2: Handshake
        auth_success, extra_nonce = self.test_handshake()
        if not auth_success:
            print("[FAIL] Handshake test failed")
            return False
            
        # Test 3: Job fetching
        job_data, extra_nonce = self.test_job_fetching()
        if not job_data:
            print("[FAIL] Job fetching test failed")
            return False
            
        # Test 4: Share submission
        if not self.test_share_submission(job_data, extra_nonce):
            print("[FAIL] Share submission test failed")
            
        # Test 5: Current system
        if not self.test_with_current_system():
            print("[FAIL] Current system test failed")
            
        print("\n" + "=" * 60)
        print("RAVENMINER FOCUS TEST COMPLETE")
        print("=" * 60)
        
        return True

if __name__ == "__main__":
    test = RavenminerFocusTest()
    test.run_complete_test() 