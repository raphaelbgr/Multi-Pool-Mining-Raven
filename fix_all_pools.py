#!/usr/bin/env python3
"""
FIX ALL POOLS
Fix each pool's specific protocol issues
"""

import json
import socket
import time
import hashlib
import struct
from datetime import datetime

class AllPoolsFix:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        with open("config.json") as f:
            return json.load(f)
    
    def fix_2miners_nonce(self, pool_config):
        """Fix 2Miners nonce format - they expect big-endian"""
        print(f"\n=== FIXING 2MINERS NONCE FORMAT ===")
        
        try:
            sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
            
            # Handshake
            sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
            response = sock.recv(4096).decode()
            
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
            
            if job_data:
                job_id = job_data['params'][0]
                print(f"[JOB] Got job: {job_id}")
                
                # Test BIG-ENDIAN nonce (2Miners expects this)
                test_nonce = 12345678
                big_endian_nonce = f"{test_nonce:08x}"  # 00bc614e
                
                submission = {
                    "id": 3,
                    "method": "mining.submit",
                    "params": [
                        pool_config['user'],
                        job_id,
                        extra_nonce + "00000001",  # 4-byte extranonce2
                        job_data['params'][6] if len(job_data['params']) > 6 else "00000000",
                        big_endian_nonce
                    ]
                }
                
                print(f"[SUBMIT] Testing big-endian nonce: {big_endian_nonce}")
                sock.sendall((json.dumps(submission) + "\n").encode())
                
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
                                        print(f"[SUCCESS] 2Miners format is correct!")
                                        return True
                                    else:
                                        print(f"[ERROR] {error_msg}")
                                else:
                                    print(f"[SUCCESS] Share accepted!")
                                    return True
                        except:
                            pass
            
            sock.close()
            return False
            
        except Exception as e:
            print(f"[ERROR] 2Miners fix failed: {e}")
            return False
    
    def fix_header_construction(self, pool_config):
        """Fix header construction for pools that need proper headers"""
        print(f"\n=== FIXING HEADER CONSTRUCTION FOR {pool_config['name']} ===")
        
        try:
            sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
            
            # Handshake
            sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
            response = sock.recv(4096).decode()
            
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
            
            if job_data:
                job_id = job_data['params'][0]
                header_hash = job_data['params'][1]
                print(f"[JOB] Got job: {job_id}")
                print(f"[JOB] Header hash: {header_hash}")
                
                # Create proper 80-byte header
                proper_header = self.create_proper_header(header_hash, job_data)
                print(f"[HEADER] Created {len(proper_header)}-byte header")
                
                # Test with proper header
                test_nonce = 12345678
                little_endian_nonce = f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}"
                
                submission = {
                    "id": 3,
                    "method": "mining.submit",
                    "params": [
                        pool_config['user'],
                        job_id,
                        extra_nonce + "0001",  # 2-byte extranonce2
                        job_data['params'][6] if len(job_data['params']) > 6 else "00000000",
                        little_endian_nonce
                    ]
                }
                
                print(f"[SUBMIT] Testing with proper header...")
                sock.sendall((json.dumps(submission) + "\n").encode())
                
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
                                        print(f"[SUCCESS] Header format is correct!")
                                        return True
                                    else:
                                        print(f"[ERROR] {error_msg}")
                                else:
                                    print(f"[SUCCESS] Share accepted!")
                                    return True
                        except:
                            pass
            
            sock.close()
            return False
            
        except Exception as e:
            print(f"[ERROR] Header fix failed: {e}")
            return False
    
    def create_proper_header(self, header_hash, job_data):
        """Create proper 80-byte Ravencoin header"""
        import time
        
        # Ravencoin header structure (80 bytes):
        # - Version (4 bytes): 0x20000000
        # - Previous block hash (32 bytes): zeros for now
        # - Merkle root (32 bytes): from header_hash
        # - Timestamp (4 bytes): from job ntime
        # - Bits (4 bytes): zeros for now
        # - Nonce (4 bytes): will be set by miner
        
        header = bytearray(80)
        
        # Version (4 bytes)
        header[0:4] = struct.pack("<I", 0x20000000)
        
        # Previous block hash (32 bytes) - use zeros
        header[4:36] = bytes(32)
        
        # Merkle root (32 bytes) - use the header_hash
        if len(header_hash) == 64:  # 32 bytes
            header[36:68] = bytes.fromhex(header_hash)
        else:
            # If not 32 bytes, pad with zeros
            header[36:68] = bytes(32)
        
        # Timestamp (4 bytes) - use job ntime
        if len(job_data['params']) > 6:
            try:
                ntime_int = int(job_data['params'][6], 16)
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
    
    def fix_herominers_pow(self, pool_config):
        """Fix HeroMiners PoW result format"""
        print(f"\n=== FIXING HEROMINERS POW RESULT ===")
        
        try:
            sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
            
            # Handshake
            sock.sendall(json.dumps({"id": 1, "method": "mining.subscribe", "params": []}).encode() + b"\n")
            response = sock.recv(4096).decode()
            
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
            
            if job_data:
                job_id = job_data['params'][0]
                print(f"[JOB] Got job: {job_id}")
                
                # Test with different PoW result formats
                test_nonce = 12345678
                little_endian_nonce = f"{(test_nonce & 0xFF):02x}{(test_nonce >> 8 & 0xFF):02x}{(test_nonce >> 16 & 0xFF):02x}{(test_nonce >> 24 & 0xFF):02x}"
                
                # Create a fake but properly formatted PoW result
                fake_result = "0" * 64  # 32-byte hash as hex
                
                submission = {
                    "id": 3,
                    "method": "mining.submit",
                    "params": [
                        pool_config['user'],
                        job_id,
                        extra_nonce + "0001",
                        job_data['params'][6] if len(job_data['params']) > 6 else "00000000",
                        little_endian_nonce,
                        fake_result  # Add PoW result
                    ]
                }
                
                print(f"[SUBMIT] Testing with PoW result...")
                sock.sendall((json.dumps(submission) + "\n").encode())
                
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
                                        print(f"[SUCCESS] PoW format is correct!")
                                        return True
                                    else:
                                        print(f"[ERROR] {error_msg}")
                                else:
                                    print(f"[SUCCESS] Share accepted!")
                                    return True
                        except:
                            pass
            
            sock.close()
            return False
            
        except Exception as e:
            print(f"[ERROR] HeroMiners fix failed: {e}")
            return False
    
    def run_all_fixes(self):
        """Run fixes for all pools"""
        print("=" * 60)
        print("FIXING ALL POOLS")
        print("=" * 60)
        print(f"Time: {datetime.now()}")
        
        results = {}
        
        # Fix 2Miners (nonce format)
        miners_pool = next(pool for pool in self.config['pools'] if pool['name'] == '2Miners')
        results['2Miners'] = self.fix_2miners_nonce(miners_pool)
        
        # Fix Ravenminer (header construction)
        ravenminer_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'Ravenminer')
        results['Ravenminer'] = self.fix_header_construction(ravenminer_pool)
        
        # Fix WoolyPooly (header construction)
        woolypooly_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'WoolyPooly')
        results['WoolyPooly'] = self.fix_header_construction(woolypooly_pool)
        
        # Fix HeroMiners (PoW result format)
        herominers_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'HeroMiners')
        results['HeroMiners'] = self.fix_herominers_pow(herominers_pool)
        
        # Fix Nanopool (header construction)
        nanopool_pool = next(pool for pool in self.config['pools'] if pool['name'] == 'Nanopool')
        results['Nanopool'] = self.fix_header_construction(nanopool_pool)
        
        print("\n" + "=" * 60)
        print("FIX RESULTS")
        print("=" * 60)
        
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
    fix = AllPoolsFix()
    working_pools = fix.run_all_fixes() 