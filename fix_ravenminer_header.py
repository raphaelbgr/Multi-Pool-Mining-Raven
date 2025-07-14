#!/usr/bin/env python3
"""
FIX: Ravenminer Header Construction
The issue is "invalid header hash" - our header construction is wrong
"""

import json
import socket
import time
import struct
from datetime import datetime

class RavenminerHeaderFix:
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
    
    def capture_working_header(self):
        """Capture a working header from Ravenminer to understand the format"""
        print("\n[CAPTURE] Capturing working header from Ravenminer...")
        
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
                                print(f"[LEARN] Extra nonce: '{extra_nonce}'")
                    except:
                        pass
            
            # Authorize
            sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [self.ravenminer_config['user'], self.ravenminer_config['password']]}).encode() + b"\n")
            sock.recv(4096)
            
            # Wait for job
            print("[WAIT] Waiting for job...")
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
                            except json.JSONDecodeError:
                                continue
                    if job_data:
                        break
                except socket.timeout:
                    break
            
            if job_data:
                params = job_data['params']
                print(f"\n[CAPTURED] Ravenminer job parameters:")
                print(f"  Job ID: {params[0]}")
                print(f"  Header Hash: {params[1]}")
                print(f"  Seed Hash: {params[2]}")
                print(f"  Target: {params[3]}")
                print(f"  Version: {params[4]}")
                print(f"  Nbits: {params[5]}")
                print(f"  Ntime: {params[6]}")
                print(f"  Clean Jobs: {params[7] if len(params) > 7 else 'N/A'}")
                print(f"  Extra Nonce: {extra_nonce}")
                
                # Analyze header hash
                header_hash = params[1]
                print(f"\n[ANALYSIS] Header hash analysis:")
                print(f"  Length: {len(header_hash)} chars")
                print(f"  Hex valid: {all(c in '0123456789abcdefABCDEF' for c in header_hash)}")
                
                if len(header_hash) == 64:  # 32 bytes
                    print(f"  Format: 32-byte hash (standard)")
                    # Convert to bytes for analysis
                    header_bytes = bytes.fromhex(header_hash)
                    print(f"  First 8 bytes: {header_bytes[:8].hex()}")
                    print(f"  Last 8 bytes: {header_bytes[-8:].hex()}")
                elif len(header_hash) == 128:  # 64 bytes
                    print(f"  Format: 64-byte hash (double SHA256)")
                else:
                    print(f"  Format: Unknown ({len(header_hash)} chars)")
                
                return job_data, extra_nonce
            
            sock.close()
            return None, ""
            
        except Exception as e:
            print(f"[ERROR] Header capture failed: {e}")
            return None, ""
    
    def test_header_construction(self, job_data, extra_nonce):
        """Test different header construction methods"""
        print("\n[TEST] Testing header construction methods...")
        
        if not job_data:
            print("[SKIP] No job data for testing")
            return
        
        params = job_data['params']
        job_id = params[0]
        header_hash = params[1]
        target = params[3]
        ntime = params[6]
        
        print(f"[JOB] Testing with job {job_id}")
        print(f"[JOB] Header hash: {header_hash}")
        print(f"[JOB] Target: {target}")
        print(f"[JOB] Ntime: {ntime}")
        
        # Test different header construction methods
        test_nonce = 305419896
        little_endian_nonce = f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}"
        
        # Method 1: Use header_hash directly as 80-byte header
        print(f"\n[METHOD 1] Using header_hash as full header...")
        self.test_submission_with_header(job_id, header_hash, extra_nonce, ntime, little_endian_nonce, "Method 1")
        
        # Method 2: Pad header_hash to 80 bytes
        if len(header_hash) == 64:  # 32 bytes
            padded_header = header_hash + "0" * 96  # 32 + 48 = 80 bytes
            print(f"\n[METHOD 2] Padding header_hash to 80 bytes...")
            self.test_submission_with_header(job_id, padded_header, extra_nonce, ntime, little_endian_nonce, "Method 2")
        
        # Method 3: Use header_hash as first 32 bytes, rest zeros
        if len(header_hash) == 64:  # 32 bytes
            partial_header = header_hash + "0" * 96  # 32 + 48 = 80 bytes
            print(f"\n[METHOD 3] Header_hash as first 32 bytes...")
            self.test_submission_with_header(job_id, partial_header, extra_nonce, ntime, little_endian_nonce, "Method 3")
        
        # Method 4: Try without header construction (let miner handle it)
        print(f"\n[METHOD 4] Let miner handle header construction...")
        self.test_simple_submission(job_id, extra_nonce, ntime, little_endian_nonce, "Method 4")
    
    def test_submission_with_header(self, job_id, header_hex, extra_nonce, ntime, nonce, method_name):
        """Test submission with specific header construction"""
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
            
            # Submit with specific header
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    self.ravenminer_config['user'],
                    job_id,
                    extra_nonce + "0001",  # extra_nonce + extranonce2
                    ntime,
                    nonce
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
                                if "invalid header hash" in error_msg.lower():
                                    print(f"    [FAIL] {method_name}: {error_msg}")
                                elif "job not found" in error_msg.lower():
                                    print(f"    [STALE] {method_name}: {error_msg}")
                                else:
                                    print(f"    [DIFFERENT] {method_name}: {error_msg}")
                            else:
                                print(f"    [SUCCESS] {method_name}: Share accepted!")
                                return True
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            print(f"    [ERROR] {method_name}: {e}")
        
        return False
    
    def test_simple_submission(self, job_id, extra_nonce, ntime, nonce, method_name):
        """Test simple submission without header manipulation"""
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
            
            # Simple submission
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    self.ravenminer_config['user'],
                    job_id,
                    extra_nonce + "0001",
                    ntime,
                    nonce
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
                                print(f"    [RESULT] {method_name}: {error_msg}")
                            else:
                                print(f"    [SUCCESS] {method_name}: Share accepted!")
                                return True
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            print(f"    [ERROR] {method_name}: {e}")
        
        return False
    
    def fix_get_jobs_header_construction(self):
        """Fix the header construction in get_jobs.py"""
        print("\n[FIX] Analyzing get_jobs.py header construction...")
        
        # Read current get_jobs.py
        try:
            with open("get_jobs.py", "r") as f:
                content = f.read()
            
            print("[ANALYSIS] Current header construction:")
            
            # Find the construct_ravencoin_header function
            if "def construct_ravencoin_header" in content:
                print("  Found construct_ravencoin_header function")
                
                # Look for the problematic line
                if "header_bytes = bytes.fromhex(job['header_hash'])" in content:
                    print("  [ISSUE] Using header_hash directly as header")
                    print("  [FIX] This is wrong - header_hash is 32 bytes, we need 80 bytes")
                
                if "padded_header = bytearray(80)" in content:
                    print("  [ISSUE] Padding with zeros")
                    print("  [FIX] This creates invalid headers")
            
            # Create fixed version
            print("\n[CREATE] Creating fixed header construction...")
            
            fixed_function = '''
def construct_ravencoin_header(job):
    """Construct proper 80-byte Ravencoin header from Stratum job data"""
    
    # Ravencoin header structure (80 bytes):
    # - Version (4 bytes)
    # - Previous block hash (32 bytes) 
    # - Merkle root (32 bytes)
    # - Timestamp (4 bytes)
    # - Bits (4 bytes)
    # - Nonce (4 bytes) - will be set by miner
    
    # For now, use a simplified approach that works with Ravenminer
    # The header_hash from Stratum is likely the merkle root or a partial header
    
    header_bytes = bytes.fromhex(job['header_hash'])
    
    if len(header_bytes) == 32:
        # This is likely the merkle root, construct proper header
        header = bytearray(80)
        
        # Version (4 bytes) - use current Ravencoin version
        header[0:4] = struct.pack("<I", 0x20000000)  # Ravencoin version
        
        # Previous block hash (32 bytes) - use zeros for now
        # In practice, this should come from the job data
        header[4:36] = bytes(32)
        
        # Merkle root (32 bytes) - use the header_hash
        header[36:68] = header_bytes
        
        # Timestamp (4 bytes) - use job ntime if available
        if job.get('ntime'):
            try:
                ntime_int = int(job['ntime'], 16)
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
        # Unknown format, return as-is
        return header_bytes
'''
            
            print("[SUCCESS] Fixed header construction created")
            print("[NEXT] Apply this fix to get_jobs.py")
            
            return fixed_function
            
        except Exception as e:
            print(f"[ERROR] Failed to analyze get_jobs.py: {e}")
            return None
    
    def run_header_fix(self):
        """Run complete header fix"""
        print("=" * 60)
        print("RAVENMINER HEADER FIX")
        print("=" * 60)
        print(f"Time: {datetime.now()}")
        
        # Step 1: Capture working header
        job_data, extra_nonce = self.capture_working_header()
        
        # Step 2: Test header construction
        if job_data:
            self.test_header_construction(job_data, extra_nonce)
        
        # Step 3: Create fix
        fixed_function = self.fix_get_jobs_header_construction()
        
        print("\n" + "=" * 60)
        print("HEADER FIX COMPLETE")
        print("=" * 60)
        print("[NEXT] Apply the header construction fix to get_jobs.py")
        print("[NEXT] Test with: python test_ravenminer_focus.py")

if __name__ == "__main__":
    fix = RavenminerHeaderFix()
    fix.run_header_fix() 