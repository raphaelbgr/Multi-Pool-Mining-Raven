#!/usr/bin/env python3
"""
REAL-TIME RAVENMINER SOLUTION
Submit shares immediately when found to avoid job staleness
"""

import json
import socket
import time
import subprocess
import threading
from datetime import datetime

class RealtimeRavenminerMiner:
    def __init__(self):
        self.config = self.load_config()
        self.ravenminer_config = None
        
        for pool in self.config['pools']:
            if pool['name'] == 'Ravenminer':
                self.ravenminer_config = pool
                break
        
        self.current_job = None
        self.extra_nonce = ""
        self.socket = None
        self.mining = False
        self.job_age = 0
        
    def load_config(self):
        with open("config.json") as f:
            return json.load(f)
    
    def connect_and_listen(self):
        """Connect to Ravenminer and listen for jobs in real-time"""
        print("[CONNECT] Connecting to Ravenminer...")
        
        try:
            self.socket = socket.create_connection(
                (self.ravenminer_config['host'], self.ravenminer_config['port']), 
                timeout=10
            )
            
            # Subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            self.socket.sendall((json.dumps(subscribe_msg) + "\n").encode())
            response = self.socket.recv(4096).decode()
            
            # Parse extra_nonce
            for line in response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 1:
                            result = parsed.get('result', [])
                            if len(result) >= 2:
                                self.extra_nonce = result[1] or ""
                                print(f"[LEARN] Extra nonce: '{self.extra_nonce}'")
                    except:
                        pass
            
            # Authorize
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [self.ravenminer_config['user'], self.ravenminer_config['password']]
            }
            self.socket.sendall((json.dumps(auth_msg) + "\n").encode())
            auth_response = self.socket.recv(4096).decode()
            
            # Parse auth result
            auth_success = False
            for line in auth_response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 2:
                            auth_success = parsed.get('result', False)
                            break
                    except:
                        pass
            
            if not auth_success:
                print("[ERROR] Authorization failed")
                return False
            
            print("[OK] Connected and authorized")
            return True
            
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False
    
    def listen_for_jobs(self):
        """Listen for new jobs and start mining immediately"""
        print("[LISTEN] Listening for jobs in real-time...")
        
        while self.mining:
            try:
                data = self.socket.recv(4096).decode()
                for line in data.split('\n'):
                    if line.strip():
                        try:
                            parsed = json.loads(line)
                            if parsed.get('method') == 'mining.notify':
                                self.current_job = parsed
                                job_id = parsed['params'][0]
                                print(f"[JOB] New job received: {job_id} at {datetime.now()}")
                                
                                # Start mining immediately with this fresh job
                                self.mine_with_fresh_job(parsed)
                        except json.JSONDecodeError:
                            continue
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[ERROR] Job listening error: {e}")
                break
    
    def mine_with_fresh_job(self, job_data):
        """Mine with a fresh job and submit immediately when found"""
        print(f"[MINE] Starting mining with fresh job {job_data['params'][0]}")
        
        # Create mining job
        self.create_mining_job(job_data)
        
        # Run miner with very short timeout for fresh jobs
        try:
            result = subprocess.run(
                ["./miner_multi_target.exe", "0"],
                capture_output=True,
                text=True,
                timeout=5  # Very short timeout for fresh jobs
            )
            
            if result.returncode == 0:
                # Check for valid nonces immediately
                for line in result.stdout.splitlines():
                    if "Valid nonce found for pool" in line:
                        # Extract nonce
                        import re
                        match = re.search(r'Valid nonce found for pool (\d+): (\d+)', line)
                        if match:
                            pool_idx = int(match.group(1))
                            nonce = int(match.group(2))
                            
                            if pool_idx == 1:  # Ravenminer is pool 1
                                print(f"[FOUND] Valid nonce: {nonce} - SUBMITTING IMMEDIATELY!")
                                self.submit_share_immediately(nonce, job_data)
                                return True
            else:
                print(f"[MINER] Miner failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("[MINER] Mining timed out (5s) - job may be stale")
        except Exception as e:
            print(f"[ERROR] Mining error: {e}")
        
        return False
    
    def create_mining_job(self, job_data):
        """Create mining job file for the CUDA miner"""
        params = job_data['params']
        
        # Create job data
        job = {
            'pool_index': 1,  # Ravenminer
            'pool_name': 'Ravenminer',
            'job_id': params[0],
            'header_hash': params[1],
            'target': params[3],
            'ntime': params[6] if len(params) > 6 else "00000000"
        }
        
        # Save job
        with open("jobs.json", "w") as f:
            json.dump([job], f, indent=2)
        
        # Create headers.bin
        self.create_headers_bin([job])
        
        print(f"[JOB] Created mining job: {job['job_id']}")
    
    def create_headers_bin(self, jobs):
        """Create headers.bin file for the miner"""
        import struct
        
        headers = bytearray()
        targets = bytearray()
        
        # Create slot for Ravenminer (pool 1)
        job_dict = {job['pool_index']: job for job in jobs}
        
        for i in range(5):  # 5 pools total
            if i in job_dict:
                job = job_dict[i]
                
                # Construct proper 80-byte header
                full_header = self.construct_ravencoin_header(job)
                
                headers.extend(struct.pack("<I", len(full_header)))  # Should be 80
                headers.extend(full_header)
                targets.extend(bytes.fromhex(job['target']))
            else:
                # Empty slot
                headers.extend(struct.pack("<I", 80))
                headers.extend(bytes(80))
                targets.extend(bytes(32))
        
        # Save to headers.bin
        with open("headers.bin", "wb") as f:
            f.write(headers + targets)
    
    def construct_ravencoin_header(self, job):
        """Construct proper 80-byte Ravencoin header"""
        import struct
        import time
        
        header_bytes = bytes.fromhex(job['header_hash'])
        
        if len(header_bytes) == 32:
            # Construct proper header
            header = bytearray(80)
            
            # Version (4 bytes)
            header[0:4] = struct.pack("<I", 0x20000000)
            
            # Previous block hash (32 bytes) - use zeros
            header[4:36] = bytes(32)
            
            # Merkle root (32 bytes) - use the header_hash
            header[36:68] = header_bytes
            
            # Timestamp (4 bytes)
            if job.get('ntime'):
                try:
                    ntime_int = int(job['ntime'], 16)
                    header[68:72] = struct.pack("<I", ntime_int)
                except:
                    header[68:72] = struct.pack("<I", int(time.time()))
            else:
                header[68:72] = struct.pack("<I", int(time.time()))
            
            # Bits (4 bytes)
            header[72:76] = bytes(4)
            
            # Nonce (4 bytes) - will be set by miner
            header[76:80] = bytes(4)
            
            return bytes(header)
        else:
            return header_bytes
    
    def submit_share_immediately(self, nonce, job_data):
        """Submit share immediately to Ravenminer"""
        print(f"[SUBMIT] Submitting nonce {nonce} IMMEDIATELY to fresh job...")
        
        try:
            params = job_data['params']
            job_id = params[0]
            ntime = params[6] if len(params) > 6 else "00000000"
            
            # Format nonce in little-endian
            little_endian_nonce = f"{(nonce & 0xFF):02x}{(nonce >> 8 & 0xFF):02x}{(nonce >> 16 & 0xFF):02x}{(nonce >> 24 & 0xFF):02x}"
            
            # Submit share IMMEDIATELY
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    self.ravenminer_config['user'],
                    job_id,
                    self.extra_nonce + "0001",  # extra_nonce + extranonce2
                    ntime,
                    little_endian_nonce
                ]
            }
            
            print(f"[SUBMIT] Sending: {json.dumps(submission)}")
            self.socket.sendall((json.dumps(submission) + "\n").encode())
            
            # Get response immediately
            response = self.socket.recv(2048).decode()
            print(f"[RESPONSE] {response.strip()}")
            
            # Parse response
            for line in response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 3:
                            error = parsed.get('error')
                            if error:
                                error_msg = error[1] if isinstance(error, list) and len(error) > 1 else str(error)
                                if "job not found" in error_msg.lower():
                                    print(f"[STALE] Job expired: {error_msg}")
                                else:
                                    print(f"[ERROR] {error_msg}")
                            else:
                                print(f"[SUCCESS] Share accepted!")
                                return True
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            print(f"[ERROR] Share submission failed: {e}")
        
        return False
    
    def start_realtime_mining(self):
        """Start real-time mining with immediate submission"""
        print("=" * 60)
        print("REAL-TIME RAVENMINER MINER")
        print("=" * 60)
        print(f"Time: {datetime.now()}")
        print("[STRATEGY] Submit shares immediately when found to avoid job staleness")
        
        # Connect to Ravenminer
        if not self.connect_and_listen():
            return False
        
        # Start job listening in background
        self.mining = True
        job_thread = threading.Thread(target=self.listen_for_jobs)
        job_thread.daemon = True
        job_thread.start()
        
        print("[START] Real-time mining started!")
        print("[INFO] Listening for jobs and mining in real-time...")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[STOP] Stopping real-time miner...")
            self.mining = False
            if self.socket:
                self.socket.close()
        
        return True

if __name__ == "__main__":
    miner = RealtimeRavenminerMiner()
    miner.start_realtime_mining() 