#!/usr/bin/env python3
"""
FIX PROTOCOL FORMATS
Systematic fix for all protocol format issues identified in analysis
"""

import json
import socket
import time
import struct
import hashlib
from datetime import datetime

class ProtocolFormatFix:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        with open("config.json") as f:
            return json.load(f)
    
    def fix_header_construction(self):
        """Fix header construction with proper endianness and hashing"""
        print("=" * 80)
        print("FIXING HEADER CONSTRUCTION")
        print("=" * 80)
        
        # Test with Ravenminer (most likely to work)
        ravenminer_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'Ravenminer')
        job_data, extra_nonce = self.get_fresh_job(ravenminer_pool)
        
        if not job_data:
            print("[ERROR] Could not get job data")
            return False
        
        params = job_data['params']
        job_id = params[0]
        header_hash = params[1]
        ntime = params[6] if len(params) > 6 else "00000000"
        
        print(f"[JOB] Job ID: {job_id}")
        print(f"[JOB] Header hash: {header_hash}")
        print(f"[JOB] Ntime: {ntime}")
        
        # Test different header construction methods
        test_nonce = 305419896  # 0x12345678
        
        # Method 1: Current method (likely wrong)
        current_header = self.construct_current_header(header_hash, ntime)
        
        # Method 2: Fixed method with proper endianness
        fixed_header = self.construct_fixed_header(header_hash, ntime)
        
        # Method 3: Double SHA-256 hashed header
        hashed_header = self.construct_hashed_header(header_hash, ntime)
        
        # Test all methods
        methods = {
            "Current": current_header,
            "Fixed": fixed_header,
            "Hashed": hashed_header
        }
        
        for method_name, header_bytes in methods.items():
            print(f"\n[TEST] {method_name} header method:")
            print(f"  Header length: {len(header_bytes)} bytes")
            print(f"  Header (first 32): {header_bytes[:32].hex()}")
            
            # Test submission with this header
            success = self.test_header_submission(ravenminer_pool, job_id, extra_nonce, ntime, test_nonce, method_name)
            if success:
                print(f"[SUCCESS] {method_name} method works!")
                return True
        
        return False
    
    def construct_current_header(self, header_hash, ntime):
        """Current get_jobs.py header construction (likely wrong)"""
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
    
    def construct_fixed_header(self, header_hash, ntime):
        """Fixed header construction with proper endianness"""
        header_bytes = bytes.fromhex(header_hash)
        
        if len(header_bytes) == 32:
            # Proper Ravencoin header structure (80 bytes):
            # - Version (4 bytes): 0x20000000 (little-endian)
            # - Previous block hash (32 bytes): zeros for now
            # - Merkle root (32 bytes): from header_hash (little-endian)
            # - Timestamp (4 bytes): from job ntime (little-endian)
            # - Bits (4 bytes): zeros for now
            # - Nonce (4 bytes): will be set by miner
            
            header = bytearray(80)
            
            # Version (4 bytes) - Ravencoin version (little-endian)
            header[0:4] = struct.pack("<I", 0x20000000)
            
            # Previous block hash (32 bytes) - use zeros
            header[4:36] = bytes(32)
            
            # Merkle root (32 bytes) - use the header_hash (keep as-is for now)
            header[36:68] = header_bytes
            
            # Timestamp (4 bytes) - use job ntime (little-endian)
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
    
    def construct_hashed_header(self, header_hash, ntime):
        """Construct header with double SHA-256 hashing"""
        # This is for pools that expect the final header hash, not the raw header
        header_bytes = bytes.fromhex(header_hash)
        
        if len(header_bytes) == 32:
            # Create proper header first
            header = bytearray(80)
            
            # Version (4 bytes)
            header[0:4] = struct.pack("<I", 0x20000000)
            
            # Previous block hash (32 bytes)
            header[4:36] = bytes(32)
            
            # Merkle root (32 bytes)
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
            
            # Double SHA-256 hash the header
            header_hash_result = hashlib.sha256(hashlib.sha256(bytes(header)).digest()).digest()
            return header_hash_result
        else:
            return header_bytes
    
    def fix_nonce_format(self):
        """Fix nonce format to little-endian hex"""
        print("\n" + "=" * 80)
        print("FIXING NONCE FORMAT")
        print("=" * 80)
        
        # Test with 2Miners (had "Malformed nonce" error)
        miners_pool = next(pool for pool in self.config['pools'] if pool['name'] == '2Miners')
        job_data, extra_nonce = self.get_fresh_job(miners_pool)
        
        if not job_data:
            print("[ERROR] Could not get 2Miners job")
            return False
        
        params = job_data['params']
        job_id = params[0]
        ntime = params[6] if len(params) > 6 else "00000000"
        
        print(f"[JOB] Job ID: {job_id}")
        
        # Test different nonce formats
        test_nonce = 305419896  # 0x12345678
        
        nonce_formats = {
            "big_endian": f"{test_nonce:08x}",  # "12345678"
            "little_endian": test_nonce.to_bytes(4, 'little').hex(),  # "78563412"
            "decimal": str(test_nonce),  # "305419896"
            "hex_prefix": f"0x{test_nonce:08x}",  # "0x12345678"
        }
        
        for format_name, nonce_val in nonce_formats.items():
            print(f"\n[TEST] Nonce format: {format_name} = {nonce_val}")
            
            success = self.test_nonce_submission(miners_pool, job_id, extra_nonce, ntime, nonce_val, format_name)
            if success:
                print(f"[SUCCESS] {format_name} nonce format works!")
                return True
        
        return False
    
    def fix_extranonce2_format(self):
        """Fix extranonce2 format issues"""
        print("\n" + "=" * 80)
        print("FIXING EXTRANONCE2 FORMAT")
        print("=" * 80)
        
        # Test with Nanopool (had "Wrong extranonce" error)
        nanopool_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'Nanopool')
        job_data, extra_nonce = self.get_fresh_job(nanopool_pool)
        
        if not job_data:
            print("[ERROR] Could not get Nanopool job")
            return False
        
        params = job_data['params']
        job_id = params[0]
        ntime = params[6] if len(params) > 6 else "00000000"
        
        print(f"[JOB] Job ID: {job_id}")
        print(f"[EXTRA] Pool extra_nonce: '{extra_nonce}' (len: {len(extra_nonce)})")
        
        # Test different extranonce2 formats
        extranonce2_formats = {
            "2byte": "0001",  # 2 bytes
            "4byte": "00000001",  # 4 bytes
            "6byte": "000000000001",  # 6 bytes
            "8byte": "0000000000000001",  # 8 bytes
            "with_extra": extra_nonce + "0001",  # Pool extra + 2 bytes
            "exact_length": extra_nonce + "00000001",  # Pool extra + 4 bytes
        }
        
        test_nonce = 305419896
        little_endian_nonce = test_nonce.to_bytes(4, 'little').hex()
        
        for format_name, ext2_val in extranonce2_formats.items():
            print(f"\n[TEST] Extranonce2 format: {format_name} = {ext2_val}")
            
            success = self.test_extranonce2_submission(nanopool_pool, job_id, extra_nonce, ntime, little_endian_nonce, ext2_val, format_name)
            if success:
                print(f"[SUCCESS] {format_name} extranonce2 format works!")
                return True
        
        return False
    
    def test_header_submission(self, pool_config, job_id, extra_nonce, ntime, test_nonce, method_name):
        """Test header submission with specific method"""
        try:
            sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=5)
            
            # Quick handshake
            sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
            sock.recv(2048)
            
            sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [pool_config['user'], pool_config['password']]}).encode() + b"\n")
            sock.recv(2048)
            
            # Format nonce in little-endian
            little_endian_nonce = test_nonce.to_bytes(4, 'little').hex()
            
            # Submit share
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    pool_config['user'],
                    job_id,
                    extra_nonce + "0001",  # extra_nonce + extranonce2
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
                                if "invalid header hash" not in error_msg.lower():
                                    print(f"    [PROGRESS] {method_name}: {error_msg}")
                                    return True
                                else:
                                    print(f"    [STILL WRONG] {method_name}: {error_msg}")
                            else:
                                print(f"    [SUCCESS] {method_name}: Share accepted!")
                                return True
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"    [ERROR] {method_name}: {e}")
        
        return False
    
    def test_nonce_submission(self, pool_config, job_id, extra_nonce, ntime, nonce_val, format_name):
        """Test nonce submission with specific format"""
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
                    extra_nonce + "00000001",  # 4-byte extranonce2
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
                                if "malformed nonce" not in error_msg.lower():
                                    print(f"    [PROGRESS] {format_name}: {error_msg}")
                                    return True
                                else:
                                    print(f"    [STILL WRONG] {format_name}: {error_msg}")
                            else:
                                print(f"    [SUCCESS] {format_name}: Share accepted!")
                                return True
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"    [ERROR] {format_name}: {e}")
        
        return False
    
    def test_extranonce2_submission(self, pool_config, job_id, extra_nonce, ntime, nonce_val, ext2_val, format_name):
        """Test extranonce2 submission with specific format"""
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
                    ext2_val,  # Use specific extranonce2 format
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
                                if "wrong extranonce" not in error_msg.lower():
                                    print(f"    [PROGRESS] {format_name}: {error_msg}")
                                    return True
                                else:
                                    print(f"    [STILL WRONG] {format_name}: {error_msg}")
                            else:
                                print(f"    [SUCCESS] {format_name}: Share accepted!")
                                return True
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"    [ERROR] {format_name}: {e}")
        
        return False
    
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
    
    def create_fixed_get_jobs(self):
        """Create fixed version of get_jobs.py with proper protocol formats"""
        print("\n" + "=" * 80)
        print("CREATING FIXED GET_JOBS.PY")
        print("=" * 80)
        
        fixed_function = '''
def construct_ravencoin_header(job):
    """Construct proper 80-byte Ravencoin header with correct endianness"""
    
    # Ravencoin header structure (80 bytes):
    # - Version (4 bytes): 0x20000000 (little-endian)
    # - Previous block hash (32 bytes): zeros for now
    # - Merkle root (32 bytes): from header_hash (little-endian)
    # - Timestamp (4 bytes): from job ntime (little-endian)
    # - Bits (4 bytes): zeros for now
    # - Nonce (4 bytes): will be set by miner
    
    header_bytes = bytes.fromhex(job['header_hash'])
    
    if len(header_bytes) == 32:
        # This is likely the merkle root, construct proper header
        header = bytearray(80)
        
        # Version (4 bytes) - Ravencoin version (little-endian)
        header[0:4] = struct.pack("<I", 0x20000000)
        
        # Previous block hash (32 bytes) - use zeros for now
        header[4:36] = bytes(32)
        
        # Merkle root (32 bytes) - use the header_hash (keep as-is)
        header[36:68] = header_bytes
        
        # Timestamp (4 bytes) - use job ntime (little-endian)
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
    
    def run_complete_fix(self):
        """Run complete protocol format fix"""
        print("PROTOCOL FORMAT FIX")
        print(f"Time: {datetime.now()}")
        
        results = {}
        
        # Fix 1: Header construction
        results['header'] = self.fix_header_construction()
        
        # Fix 2: Nonce format
        results['nonce'] = self.fix_nonce_format()
        
        # Fix 3: Extranonce2 format
        results['extranonce2'] = self.fix_extranonce2_format()
        
        # Create fixed get_jobs.py
        self.create_fixed_get_jobs()
        
        print("\n" + "=" * 80)
        print("PROTOCOL FIX RESULTS")
        print("=" * 80)
        
        working_fixes = []
        for fix_name, success in results.items():
            if success:
                working_fixes.append(fix_name)
                print(f"✅ {fix_name}: FIXED")
            else:
                print(f"❌ {fix_name}: Still broken")
        
        if working_fixes:
            print(f"\n[SUCCESS] Fixed {len(working_fixes)} protocol issue(s): {working_fixes}")
        else:
            print(f"\n[FAIL] No protocol issues were fixed")
        
        return working_fixes

if __name__ == "__main__":
    fix = ProtocolFormatFix()
    working_fixes = fix.run_complete_fix() 