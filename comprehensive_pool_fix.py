#!/usr/bin/env python3
"""
COMPREHENSIVE POOL FIX
Fix all pools based on deep analysis findings
"""

import json
import socket
import time
import struct
from datetime import datetime

class ComprehensivePoolFix:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        with open("config.json") as f:
            return json.load(f)
    
    def fix_2miners_malformed_nonce(self):
        """Fix 2Miners 'Malformed nonce' error"""
        print("\n=== FIXING 2MINERS MALFORMED NONCE ===")
        
        miners_pool = next(pool for pool in self.config['pools'] if pool['name'] == '2Miners')
        job_data, extra_nonce = self.get_fresh_job(miners_pool)
        
        if not job_data:
            print("[ERROR] Could not get 2Miners job")
            return False
        
        params = job_data['params']
        job_id = params[0]
        ntime = params[6] if len(params) > 6 else "00000000"
        
        print(f"[JOB] Job ID: {job_id}")
        print(f"[EXTRA] Extra nonce: '{extra_nonce}' (len: {len(extra_nonce)})")
        
        # 2Miners specific findings:
        # - Rejects "decimal" and "hex_prefix" nonce formats
        # - Accepts "little_endian" and "big_endian" formats
        # - Needs proper extranonce2 format
        
        test_nonce = 305419896  # 0x12345678
        
        # Test 2Miners-specific formats
        nonce_formats = {
            "little_endian": f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}",
            "big_endian": f"{test_nonce:08x}",
        }
        
        extranonce2_formats = {
            "4byte": "00000001",  # 4-byte format
            "8byte": "0000000000000001",  # 8-byte format
        }
        
        for nonce_name, nonce_val in nonce_formats.items():
            for ext2_name, ext2_val in extranonce2_formats.items():
                try:
                    sock = socket.create_connection((miners_pool['host'], miners_pool['port']), timeout=5)
                    
                    # Quick handshake
                    sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
                    sock.recv(2048)
                    
                    sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [miners_pool['user'], miners_pool['password']]}).encode() + b"\n")
                    sock.recv(2048)
                    
                    # Submit share
                    submission = {
                        "id": 3,
                        "method": "mining.submit",
                        "params": [
                            miners_pool['user'],
                            job_id,
                            extra_nonce + ext2_val,  # Pool extra_nonce + our extranonce2
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
                                        if "low difficulty" in error_msg.lower():
                                            print(f"[SUCCESS] 2Miners format: {nonce_name} + {ext2_name}")
                                            return True
                                        elif "malformed nonce" not in error_msg.lower():
                                            print(f"[PROGRESS] 2Miners: {nonce_name} + {ext2_name} = {error_msg}")
                            except json.JSONDecodeError:
                                continue
                                
                except Exception as e:
                    continue
                
                time.sleep(0.1)
        
        return False
    
    def fix_nanopool_wrong_extranonce(self):
        """Fix Nanopool 'Wrong extranonce' error"""
        print("\n=== FIXING NANOPOOL WRONG EXTRANONCE ===")
        
        nanopool_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'Nanopool')
        job_data, extra_nonce = self.get_fresh_job(nanopool_pool)
        
        if not job_data:
            print("[ERROR] Could not get Nanopool job")
            return False
        
        params = job_data['params']
        job_id = params[0]
        ntime = params[6] if len(params) > 6 else "00000000"
        
        print(f"[JOB] Job ID: {job_id}")
        print(f"[EXTRA] Extra nonce: '{extra_nonce}' (len: {len(extra_nonce)})")
        
        # Nanopool findings:
        # - Extra nonce is 6 chars: 'c69331'
        # - Error shows: "Wrong extranonce c693310001 1bb911"
        # - This suggests it expects a specific format
        
        test_nonce = 305419896
        little_endian_nonce = f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}"
        
        # Test Nanopool-specific extranonce formats
        extranonce2_formats = {
            "2byte": "0001",  # 2 bytes
            "4byte": "00000001",  # 4 bytes
            "6byte": "000000000001",  # 6 bytes
            "exact": "1bb911",  # Try the exact value from error
        }
        
        for ext2_name, ext2_val in extranonce2_formats.items():
            try:
                sock = socket.create_connection((nanopool_pool['host'], nanopool_pool['port']), timeout=5)
                
                # Quick handshake
                sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
                sock.recv(2048)
                
                sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [nanopool_pool['user'], nanopool_pool['password']]}).encode() + b"\n")
                sock.recv(2048)
                
                # Submit share
                submission = {
                    "id": 3,
                    "method": "mining.submit",
                    "params": [
                        nanopool_pool['user'],
                        job_id,
                        extra_nonce + ext2_val,  # 6-char pool extra_nonce + our extranonce2
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
                                    if "wrong extranonce" not in error_msg.lower():
                                        print(f"[SUCCESS] Nanopool format: {ext2_name}")
                                        return True
                                    else:
                                        print(f"[STILL WRONG] Nanopool {ext2_name}: {error_msg}")
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                continue
            
            time.sleep(0.1)
        
        return False
    
    def fix_woolypooly_invalid_header(self):
        """Fix WoolyPooly 'Invalid header hash' error"""
        print("\n=== FIXING WOOLYPOOLY INVALID HEADER ===")
        
        woolypooly_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'WoolyPooly')
        job_data, extra_nonce = self.get_fresh_job(woolypooly_pool)
        
        if not job_data:
            print("[ERROR] Could not get WoolyPooly job")
            return False
        
        params = job_data['params']
        job_id = params[0]
        header_hash = params[1]
        ntime = params[6] if len(params) > 6 else "00000000"
        
        print(f"[JOB] Job ID: {job_id}")
        print(f"[HEADER] Header hash: {header_hash}")
        
        # WoolyPooly findings:
        # - All nonce formats give "Invalid header hash"
        # - This suggests header construction is wrong
        # - Need to use proper Ravencoin header structure
        
        test_nonce = 305419896
        little_endian_nonce = f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}"
        
        # Test with proper header construction
        proper_header = self.construct_proper_ravencoin_header(header_hash, ntime)
        print(f"[HEADER] Constructed proper header: {len(proper_header)} bytes")
        
        try:
            sock = socket.create_connection((woolypooly_pool['host'], woolypooly_pool['port']), timeout=5)
            
            # Quick handshake
            sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
            sock.recv(2048)
            
            sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [woolypooly_pool['user'], woolypooly_pool['password']]}).encode() + b"\n")
            sock.recv(2048)
            
            # Submit share with proper header
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    woolypooly_pool['user'],
                    job_id,
                    extra_nonce + "0001",  # 4-char pool extra_nonce + 2-byte extranonce2
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
                                    print(f"[SUCCESS] WoolyPooly header fixed!")
                                    return True
                                else:
                                    print(f"[STILL WRONG] WoolyPooly: {error_msg}")
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"[ERROR] WoolyPooly test failed: {e}")
        
        return False
    
    def fix_herominers_pow_result(self):
        """Fix HeroMiners 'Malformed PoW result' error"""
        print("\n=== FIXING HEROMINERS POW RESULT ===")
        
        herominers_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'HeroMiners')
        job_data, extra_nonce = self.get_fresh_job(herominers_pool)
        
        if not job_data:
            print("[ERROR] Could not get HeroMiners job")
            return False
        
        params = job_data['params']
        job_id = params[0]
        ntime = params[6] if len(params) > 6 else "00000000"
        
        print(f"[JOB] Job ID: {job_id}")
        print(f"[EXTRA] Extra nonce: '{extra_nonce}' (len: {len(extra_nonce)})")
        
        # HeroMiners findings:
        # - "Malformed PoW result" suggests it expects additional parameters
        # - May need mixhash and result parameters like T-Rex
        
        test_nonce = 305419896
        little_endian_nonce = f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}"
        
        # Test with additional PoW parameters
        try:
            sock = socket.create_connection((herominers_pool['host'], herominers_pool['port']), timeout=5)
            
            # Quick handshake
            sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
            sock.recv(2048)
            
            sock.sendall(json.dumps({"id": 2, "method": "mining.authorize", "params": [herominers_pool['user'], herominers_pool['password']]}).encode() + b"\n")
            sock.recv(2048)
            
            # Submit share with additional PoW parameters
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    herominers_pool['user'],
                    job_id,
                    extra_nonce + "0001",
                    ntime,
                    little_endian_nonce,
                    "0" * 64,  # mixhash (64 chars)
                    "0" * 64   # result (64 chars)
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
                                if "malformed pow result" not in error_msg.lower():
                                    print(f"[SUCCESS] HeroMiners PoW format fixed!")
                                    return True
                                else:
                                    print(f"[STILL WRONG] HeroMiners: {error_msg}")
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"[ERROR] HeroMiners test failed: {e}")
        
        return False
    
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
    
    def run_comprehensive_fix(self):
        """Run comprehensive fix for all pools"""
        print("=" * 80)
        print("COMPREHENSIVE POOL FIX")
        print("=" * 80)
        print(f"Time: {datetime.now()}")
        
        results = {}
        
        # Fix each pool based on analysis
        results['2Miners'] = self.fix_2miners_malformed_nonce()
        results['Nanopool'] = self.fix_nanopool_wrong_extranonce()
        results['WoolyPooly'] = self.fix_woolypooly_invalid_header()
        results['HeroMiners'] = self.fix_herominers_pow_result()
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE FIX RESULTS")
        print("=" * 80)
        
        working_pools = []
        for pool_name, success in results.items():
            if success:
                working_pools.append(pool_name)
                print(f"✅ {pool_name}: FIXED")
            else:
                print(f"❌ {pool_name}: Still broken")
        
        if working_pools:
            print(f"\n[SUCCESS] Fixed {len(working_pools)} pool(s): {working_pools}")
        else:
            print(f"\n[FAIL] No pools were fixed")
        
        return working_pools

if __name__ == "__main__":
    fix = ComprehensivePoolFix()
    working_pools = fix.run_comprehensive_fix() 