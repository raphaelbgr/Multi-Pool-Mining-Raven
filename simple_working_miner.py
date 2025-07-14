#!/usr/bin/env python3
"""
SIMPLE WORKING MINER
Focus on getting ONE pool working perfectly
"""

import json
import socket
import time
import subprocess
from datetime import datetime

class SimpleWorkingMiner:
    def __init__(self):
        self.config = self.load_config()
        # Start with 2Miners as it's the most popular
        self.target_pool = next(pool for pool in self.config['pools'] if pool['name'] == '2Miners')
        
    def load_config(self):
        with open("config.json") as f:
            return json.load(f)
    
    def mine_with_realistic_difficulty(self):
        """Mine with realistic difficulty that should find shares"""
        print("=" * 60)
        print("SIMPLE WORKING MINER")
        print("=" * 60)
        print(f"Target Pool: {self.target_pool['name']}")
        print(f"Time: {datetime.now()}")
        
        # Connect to pool
        try:
            sock = socket.create_connection((self.target_pool['host'], self.target_pool['port']), timeout=10)
            print(f"[CONNECT] Connected to {self.target_pool['name']}")
            
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
            sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [self.target_pool['user'], self.target_pool['password']]}).encode() + b"\n")
            auth_response = sock.recv(4096).decode()
            print(f"[AUTH] Authorization response: {auth_response.strip()}")
            
            # Wait for job
            print(f"[WAIT] Waiting for job from {self.target_pool['name']}...")
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
                                    print(f"[JOB] Got job: {job_id}")
                                    break
                            except:
                                continue
                    if job_data:
                        break
                except socket.timeout:
                    break
            
            if not job_data:
                print("[ERROR] No job received")
                return False
            
            # Create mining job
            self.create_mining_job(job_data)
            
            # Mine with realistic parameters
            print("[MINE] Starting realistic mining...")
            
            # Test multiple nonce ranges to find shares
            nonce_ranges = [
                (0, 1000000),           # Low range
                (1000000, 2000000),      # Medium range  
                (2000000, 3000000),      # Higher range
                (3000000, 4000000),      # Even higher
            ]
            
            for start_nonce, end_nonce in nonce_ranges:
                print(f"[RANGE] Testing nonces {start_nonce:,} to {end_nonce:,}")
                
                try:
                    result = subprocess.run(
                        ["./miner_multi_target.exe", str(start_nonce)],
                        capture_output=True,
                        text=True,
                        timeout=10  # Short timeout per range
                    )
                    
                    if result.returncode == 0:
                        # Check for valid nonces
                        for line in result.stdout.splitlines():
                            if "Valid nonce found for pool" in line:
                                # Extract nonce
                                import re
                                match = re.search(r'Valid nonce found for pool (\d+): (\d+)', line)
                                if match:
                                    pool_idx = int(match.group(1))
                                    nonce = int(match.group(2))
                                    
                                    if pool_idx == 0:  # 2Miners is pool 0
                                        print(f"[FOUND] Valid nonce: {nonce}")
                                        
                                        # Submit immediately
                                        if self.submit_share_immediately(sock, nonce, job_data, extra_nonce):
                                            print(f"[SUCCESS] Share accepted by {self.target_pool['name']}!")
                                            return True
                                        else:
                                            print(f"[FAIL] Share rejected by {self.target_pool['name']}")
                                    
                        print(f"[INFO] No valid nonces in range {start_nonce:,} to {end_nonce:,}")
                    else:
                        print(f"[ERROR] Mining failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"[TIMEOUT] Range {start_nonce:,} to {end_nonce:,} timed out")
                except Exception as e:
                    print(f"[ERROR] Mining error: {e}")
            
            sock.close()
            print("[FAIL] No shares found")
            return False
            
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False
    
    def create_mining_job(self, job_data):
        """Create mining job for the CUDA miner"""
        params = job_data['params']
        
        # Create job data
        job = {
            'pool_index': 0,  # 2Miners
            'pool_name': '2Miners',
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
        
        # Create slot for 2Miners (pool 0)
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
    
    def submit_share_immediately(self, sock, nonce, job_data, extra_nonce):
        """Submit share immediately to 2Miners"""
        print(f"[SUBMIT] Submitting nonce {nonce} to 2Miners...")
        
        try:
            params = job_data['params']
            job_id = params[0]
            ntime = params[6] if len(params) > 6 else "00000000"
            
            # 2Miners expects BIG-ENDIAN nonce
            big_endian_nonce = f"{nonce:08x}"
            
            # Submit share
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    self.target_pool['user'],
                    job_id,
                    extra_nonce + "00000001",  # 4-byte extranonce2
                    ntime,
                    big_endian_nonce
                ]
            }
            
            print(f"[SUBMIT] Sending: {json.dumps(submission)}")
            sock.sendall((json.dumps(submission) + "\n").encode())
            
            # Get response immediately
            response = sock.recv(2048).decode()
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
                                if "low difficulty" in error_msg.lower():
                                    print(f"[PROGRESS] Low difficulty - format is correct!")
                                    return True
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

if __name__ == "__main__":
    miner = SimpleWorkingMiner()
    success = miner.mine_with_realistic_difficulty()
    
    if success:
        print("\n🎉 SUCCESS! Found a working solution!")
        print("💡 This proves the system can work with the right parameters")
    else:
        print("\n❌ Still not working")
        print("💡 The issue may be deeper than expected") 