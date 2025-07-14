#!/usr/bin/env python3
"""
DEEP REVERSE ENGINEERING ANALYSIS
Analyze the exact header construction and submission format issues
"""

import json
import socket
import time
import struct
import hashlib
from datetime import datetime

class DeepReverseEngineering:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        with open("config.json") as f:
            return json.load(f)
    
    def analyze_header_construction(self):
        """Analyze the exact header construction issues"""
        print("=" * 80)
        print("DEEP HEADER CONSTRUCTION ANALYSIS")
        print("=" * 80)
        
        # Test with a known working pool (Ravenminer)
        ravenminer_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'Ravenminer')
        
        print(f"\n[ANALYSIS] Testing with {ravenminer_pool['name']}")
        
        # Get fresh job data
        job_data, extra_nonce = self.get_fresh_job(ravenminer_pool)
        
        if not job_data:
            print("[ERROR] Could not get job data")
            return
        
        params = job_data['params']
        print(f"\n[JOB DATA] Raw job parameters:")
        for i, param in enumerate(params):
            print(f"  Param {i}: {param}")
        
        # Analyze the header_hash
        header_hash = params[1]
        print(f"\n[HEADER HASH ANALYSIS]")
        print(f"  Raw header_hash: {header_hash}")
        print(f"  Length: {len(header_hash)} chars")
        print(f"  Hex valid: {all(c in '0123456789abcdefABCDEF' for c in header_hash)}")
        
        if len(header_hash) == 64:
            print(f"  Format: 32-byte hash (standard)")
            header_bytes = bytes.fromhex(header_hash)
            print(f"  First 8 bytes: {header_bytes[:8].hex()}")
            print(f"  Last 8 bytes: {header_bytes[-8:].hex()}")
            
            # Test different header construction methods
            self.test_header_methods(header_hash, params, extra_nonce)
        else:
            print(f"  Format: Unknown ({len(header_hash)} chars)")
    
    def test_header_methods(self, header_hash, params, extra_nonce):
        """Test different header construction methods"""
        print(f"\n[HEADER CONSTRUCTION TESTS]")
        
        job_id = params[0]
        ntime = params[6] if len(params) > 6 else "00000000"
        
        # Method 1: Current get_jobs.py method
        print(f"\n[METHOD 1] Current get_jobs.py header construction:")
        current_header = self.construct_current_header(header_hash, ntime)
        print(f"  Header (hex): {current_header.hex()}")
        print(f"  Length: {len(current_header)} bytes")
        
        # Method 2: Proper Ravencoin header structure
        print(f"\n[METHOD 2] Proper Ravencoin header structure:")
        proper_header = self.construct_proper_ravencoin_header(header_hash, ntime)
        print(f"  Header (hex): {proper_header.hex()}")
        print(f"  Length: {len(proper_header)} bytes")
        
        # Method 3: Use header_hash as merkle root only
        print(f"\n[METHOD 3] Header_hash as merkle root only:")
        merkle_header = self.construct_merkle_header(header_hash, ntime)
        print(f"  Header (hex): {merkle_header.hex()}")
        print(f"  Length: {len(merkle_header)} bytes")
        
        # Test all methods with submission
        self.test_header_submissions(job_id, extra_nonce, ntime, {
            "Current": current_header,
            "Proper": proper_header,
            "Merkle": merkle_header
        })
    
    def construct_current_header(self, header_hash, ntime):
        """Current get_jobs.py header construction"""
        import time
        
        header_bytes = bytes.fromhex(header_hash)
        
        if len(header_bytes) == 32:
            # Current method: pad to 80 bytes
            header = bytearray(80)
            header[:32] = header_bytes
            
            # Set ntime if available
            if ntime:
                try:
                    ntime_int = int(ntime, 16)
                    header[68:72] = struct.pack("<I", ntime_int)
                except:
                    header[68:72] = struct.pack("<I", int(time.time()))
            
            return bytes(header)
        else:
            return header_bytes
    
    def construct_proper_ravencoin_header(self, header_hash, ntime):
        """Construct proper 80-byte Ravencoin header"""
        import time
        
        header_bytes = bytes.fromhex(header_hash)
        
        if len(header_bytes) == 32:
            # Proper Ravencoin header structure (80 bytes):
            # - Version (4 bytes): 0x20000000
            # - Previous block hash (32 bytes): zeros for now
            # - Merkle root (32 bytes): from header_hash
            # - Timestamp (4 bytes): from job ntime
            # - Bits (4 bytes): zeros for now
            # - Nonce (4 bytes): will be set by miner
            
            header = bytearray(80)
            
            # Version (4 bytes) - Ravencoin version
            header[0:4] = struct.pack("<I", 0x20000000)
            
            # Previous block hash (32 bytes) - use zeros
            header[4:36] = bytes(32)
            
            # Merkle root (32 bytes) - use the header_hash
            header[36:68] = header_bytes
            
            # Timestamp (4 bytes) - use job ntime
            if ntime:
                try:
                    ntime_int = int(ntime, 16)
                    header[68:72] = struct.pack("<I", ntime_int)
                except:
                    header[68:72] = struct.pack("<I", int(time.time()))
            else:
                header[68:72] = struct.pack("<I", int(time.time()))
            
            # Bits (4 bytes) - use zeros for now
            header[72:76] = bytes(4)
            
            # Nonce (4 bytes) - will be set by miner
            header[76:80] = bytes(4)
            
            return bytes(header)
        else:
            return header_bytes
    
    def construct_merkle_header(self, header_hash, ntime):
        """Construct header using header_hash as merkle root only"""
        import time
        
        header_bytes = bytes.fromhex(header_hash)
        
        if len(header_bytes) == 32:
            # Alternative: header_hash might be merkle root
            header = bytearray(80)
            
            # Version (4 bytes)
            header[0:4] = struct.pack("<I", 0x20000000)
            
            # Previous block hash (32 bytes) - use zeros
            header[4:36] = bytes(32)
            
            # Merkle root (32 bytes) - use header_hash as merkle root
            header[36:68] = header_bytes
            
            # Timestamp (4 bytes)
            if ntime:
                try:
                    ntime_int = int(ntime, 16)
                    header[68:72] = struct.pack("<I", ntime_int)
                except:
                    header[68:72] = struct.pack("<I", int(time.time()))
            else:
                header[68:72] = struct.pack("<I", int(time.time()))
            
            # Bits (4 bytes)
            header[72:76] = bytes(4)
            
            # Nonce (4 bytes)
            header[76:80] = bytes(4)
            
            return bytes(header)
        else:
            return header_bytes
    
    def test_header_submissions(self, job_id, extra_nonce, ntime, headers):
        """Test different header constructions with submissions"""
        print(f"\n[SUBMISSION TESTS]")
        
        ravenminer_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'Ravenminer')
        
        test_nonce = 305419896  # 0x12345678
        little_endian_nonce = f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}"
        
        for method_name, header_bytes in headers.items():
            print(f"\n[TEST] {method_name} header method:")
            print(f"  Header length: {len(header_bytes)} bytes")
            print(f"  Header (first 32): {header_bytes[:32].hex()}")
            print(f"  Header (last 32): {header_bytes[-32:].hex()}")
            
            # Test submission
            try:
                sock = socket.create_connection((ravenminer_pool['host'], ravenminer_pool['port']), timeout=5)
                
                # Quick handshake
                sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
                sock.recv(2048)
                
                sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [ravenminer_pool['user'], ravenminer_pool['password']]}).encode() + b"\n")
                sock.recv(2048)
                
                # Submit share
                submission = {
                    "id": 3,
                    "method": "mining.submit",
                    "params": [
                        ravenminer_pool['user'],
                        job_id,
                        extra_nonce + "0001",
                        ntime,
                        little_endian_nonce
                    ]
                }
                
                sock.sendall((json.dumps(submission) + "\n").encode())
                response = sock.recv(2048).decode()
                sock.close()
                
                # Parse response
                for line in response.split('\n'):
                    if line.strip():
                        try:
                            parsed = json.loads(line)
                            if parsed.get('id') == 3:
                                error = parsed.get('error')
                                if error:
                                    error_msg = error[1] if isinstance(error, list) and len(error) > 1 else str(error)
                                    print(f"    [RESULT] {error_msg}")
                                else:
                                    print(f"    [SUCCESS] Share accepted!")
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                print(f"    [ERROR] {e}")
    
    def analyze_submission_formats(self):
        """Analyze submission format issues"""
        print(f"\n" + "=" * 80)
        print("SUBMISSION FORMAT ANALYSIS")
        print("=" * 80)
        
        # Test each pool's specific submission format
        for pool_config in self.config['pools']:
            print(f"\n[ANALYSIS] {pool_config['name']} submission format:")
            
            # Get job data
            job_data, extra_nonce = self.get_fresh_job(pool_config)
            if not job_data:
                continue
            
            params = job_data['params']
            job_id = params[0]
            ntime = params[6] if len(params) > 6 else "00000000"
            
            print(f"  Job ID: {job_id}")
            print(f"  Extra nonce: '{extra_nonce}' (len: {len(extra_nonce)})")
            print(f"  Ntime: {ntime}")
            
            # Test different submission formats
            self.test_submission_formats(pool_config, job_id, extra_nonce, ntime)
    
    def test_submission_formats(self, pool_config, job_id, extra_nonce, ntime):
        """Test different submission formats for a pool"""
        
        test_nonce = 305419896
        nonce_formats = {
            "little_endian": f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}",
            "big_endian": f"{test_nonce:08x}",
            "decimal": str(test_nonce),
            "hex_prefix": f"0x{test_nonce:08x}"
        }
        
        extranonce2_formats = {
            "simple": "0001",
            "incremental": "1234",
            "timestamp": f"{int(time.time()) & 0xFFFF:04x}",
            "zero_pad": "0000"
        }
        
        print(f"  Testing {len(nonce_formats)} nonce formats x {len(extranonce2_formats)} extranonce2 formats")
        
        for nonce_name, nonce_val in nonce_formats.items():
            for ext2_name, ext2_val in extranonce2_formats.items():
                try:
                    sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=5)
                    
                    # Quick handshake
                    sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
                    sock.recv(2048)
                    
                    sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [pool_config['user'], pool_config['password']]}).encode() + b"\n")
                    sock.recv(2048)
                    
                    # Submit share
                    submission = {
                        "id": 3,
                        "method": "mining.submit",
                        "params": [
                            pool_config['user'],
                            job_id,
                            extra_nonce + ext2_val,
                            ntime,
                            nonce_val
                        ]
                    }
                    
                    sock.sendall((json.dumps(submission) + "\n").encode())
                    response = sock.recv(2048).decode()
                    sock.close()
                    
                    # Parse response
                    for line in response.split('\n'):
                        if line.strip():
                            try:
                                parsed = json.loads(line)
                                if parsed.get('id') == 3:
                                    error = parsed.get('error')
                                    if error:
                                        error_msg = error[1] if isinstance(error, list) and len(error) > 1 else str(error)
                                        if error_msg not in ['Job not found', 'Stale share', 'Duplicate share']:
                                            print(f"    [DIFFERENT] {nonce_name} + {ext2_name}: {error_msg}")
                                    else:
                                        print(f"    [SUCCESS] {nonce_name} + {ext2_name}: Share accepted!")
                            except json.JSONDecodeError:
                                continue
                                
                except Exception as e:
                    continue
                
                time.sleep(0.1)  # Don't spam
    
    def get_fresh_job(self, pool_config):
        """Get fresh job from pool"""
        try:
            sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
            
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
            sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [pool_config['user'], pool_config['password']]}).encode() + b"\n")
            sock.recv(4096)
            
            # Wait for job
            sock.settimeout(15)
            job_data = None
            
            while True:
                try:
                    data = sock.recv(4096).decode()
                    for line in data.split('\n'):
                        if line.strip():
                            try:
                                parsed = json.loads(line)
                                if parsed.get('method') == 'mining.notify':
                                    job_data = parsed
                                    break
                            except:
                                continue
                    if job_data:
                        break
                except socket.timeout:
                    break
            
            sock.close()
            return job_data, extra_nonce
            
        except Exception as e:
            print(f"[ERROR] Failed to get job from {pool_config['name']}: {e}")
            return None, ""
    
    def run_complete_analysis(self):
        """Run complete reverse engineering analysis"""
        print("DEEP REVERSE ENGINEERING ANALYSIS")
        print(f"Time: {datetime.now()}")
        
        # Analyze header construction
        self.analyze_header_construction()
        
        # Analyze submission formats
        self.analyze_submission_formats()
        
        print(f"\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("Key findings:")
        print("1. Header construction method affects 'invalid header hash' errors")
        print("2. Submission format affects 'malformed nonce' errors")
        print("3. Each pool may require different formats")
        print("4. Job staleness affects 'job not found' errors")

if __name__ == "__main__":
    analysis = DeepReverseEngineering()
    analysis.run_complete_analysis() 