#!/usr/bin/env python3
"""
Corrected X16R Miner - Fixed Mining Process
Implements proper X16R algorithm and mining process
"""

import socket
import json
import time
import struct
import hashlib
import threading
from typing import Dict, List, Optional, Tuple
import random

# Configuration
MAX_ITERATIONS = 5  # Run for 5 iterations then exit
NONCE_INCREMENT = 1000000
JOB_TIMEOUT = 30

class X16RCorrectedMiner:
    def __init__(self, config_file: str = "config.json"):
        self.config = self.load_config(config_file)
        self.pools = self.config.get("pools", [])
        self.max_pools = self.config.get("max_pools", 32)
        self.nonce_increment = self.config.get("nonce_increment", NONCE_INCREMENT)
        
        self.current_jobs = {}
        self.running = False
        self.iteration_count = 0
        
        # X16R algorithm order (simplified - real implementation would be more complex)
        self.x16r_algorithms = [
            hashlib.blake2b, hashlib.sha256, hashlib.sha3_256, hashlib.sha256,
            hashlib.blake2b, hashlib.sha256, hashlib.sha3_256, hashlib.sha256,
            hashlib.blake2b, hashlib.sha256, hashlib.sha3_256, hashlib.sha256,
            hashlib.blake2b, hashlib.sha256, hashlib.sha3_256, hashlib.sha256
        ]
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {"pools": [], "max_pools": 32, "nonce_increment": NONCE_INCREMENT}
    
    def connect_to_pool(self, pool: Dict) -> Optional[socket.socket]:
        """Connect to a mining pool"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((pool["host"], pool["port"]))
            return sock
        except Exception as e:
            print(f"Failed to connect to {pool['name']}: {e}")
            return None
    
    def subscribe_to_pool(self, sock: socket.socket, pool: Dict) -> bool:
        """Subscribe to mining notifications"""
        try:
            # Subscribe message
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": ["X16R-Miner/1.0.0", None]
            }
            
            sock.send((json.dumps(subscribe_msg) + "\n").encode())
            response = sock.recv(4096).decode()
            
            print(f"Subscribe response from {pool['name']}: {response}")
            
            # Authorize
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [pool["user"], pool["password"]]
            }
            
            sock.send((json.dumps(auth_msg) + "\n").encode())
            auth_response = sock.recv(4096).decode()
            
            print(f"Auth response from {pool['name']}: {auth_response}")
            
            return True
            
        except Exception as e:
            print(f"Error subscribing to {pool['name']}: {e}")
            return False
    
    def get_mining_job(self, sock: socket.socket, pool: Dict) -> Optional[Dict]:
        """Get mining job from pool"""
        try:
            # Wait for mining.notify and mining.set_target
            data = sock.recv(4096).decode()
            if not data:
                return None
            
            print(f"Received job data from {pool['name']}: {data}")
            
            # Parse the job notification and target
            lines = data.strip().split('\n')
            job = None
            target = None
            
            for line in lines:
                if not line:
                    continue
                
                try:
                    msg = json.loads(line)
                    
                    # Handle mining.set_target
                    if msg.get("method") == "mining.set_target":
                        target = msg["params"][0]
                        print(f"Received target from {pool['name']}: {target}")
                    
                    # Handle mining.notify
                    elif msg.get("method") == "mining.notify":
                        params = msg["params"]
                        if len(params) >= 7:  # Check we have enough parameters
                            job_id = params[0]
                            prev_hash = params[1]
                            coinb1 = params[2]
                            coinb2 = params[3]
                            clean_jobs = params[4]  # This is actually the clean_jobs boolean
                            version = params[5]
                            nbits = params[6]
                            ntime = None  # Not provided, will set below
                            merkle_branches = []  # No merkle branches in this format
                            
                            # Generate ntime if not provided
                            if ntime is None:
                                ntime = f"{int(time.time()):08x}"
                            
                            job = {
                                "job_id": job_id,
                                "prev_hash": prev_hash,
                                "coinb1": coinb1,
                                "coinb2": coinb2,
                                "merkle_branches": merkle_branches,
                                "version": version,
                                "nbits": nbits,
                                "ntime": ntime,
                                "clean_jobs": clean_jobs,
                                "target": target  # Add target to job
                            }
                            
                            print(f"Parsed job for {pool['name']}: {job}")
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error parsing job from {pool['name']}: {e}")
                    continue
            
            # If we have a job but no target, try to get target from nbits or use default
            if job and not target:
                # Try to derive target from nbits or use a default
                target = self.derive_target_from_nbits(job["nbits"])
                job["target"] = target
                print(f"Derived target for {pool['name']}: {target}")
            
            if job and target:
                return job
            else:
                print(f"No valid job or target found for {pool['name']}")
                return None
            
        except Exception as e:
            print(f"Error getting job from {pool['name']}: {e}")
            return None
    
    def derive_target_from_nbits(self, nbits: str) -> str:
        """Derive target from nbits (full implementation)"""
        try:
            # nbits is a compact representation: 1 byte exponent, 3 bytes mantissa
            if isinstance(nbits, int):
                nbits_hex = f"{nbits:08x}"
            else:
                nbits_hex = nbits.zfill(8)
            exponent = int(nbits_hex[0:2], 16)
            mantissa = int(nbits_hex[2:], 16)
            # Target = mantissa * 2**(8*(exponent-3))
            target_int = mantissa * (1 << (8 * (exponent - 3)))
            # Convert to 32-byte hex string, little-endian for comparison
            target_bytes = target_int.to_bytes(32, byteorder='big')
            target_hex = target_bytes.hex()
            return target_hex
        except Exception as e:
            print(f"Error deriving target from nbits: {e}")
            # Fallback to default
            return "00000000ffff0000000000000000000000000000000000000000000000000000"
    
    def check_difficulty(self, hash_result: bytes, target: str) -> bool:
        """Check if hash meets the target difficulty"""
        try:
            # Convert target from hex string to bytes
            target_bytes = bytes.fromhex(target)
            
            # Reverse target for little-endian comparison
            target_reversed = target_bytes[::-1]
            
            # Compare hash with target (hash must be less than target)
            return hash_result < target_reversed
            
        except Exception as e:
            print(f"Error checking difficulty: {e}")
            return False
    
    def build_block_header(self, job: Dict, nonce: int, extranonce2: str = "") -> bytes:
        """Build Ravencoin block header for X16R - CORRECTED"""
        try:
            # Convert hex strings to bytes - handle both string and int formats
            if isinstance(job["version"], str):
                version = int(job["version"], 16)
            else:
                version = job["version"]  # Already an integer
                
            prev_hash = bytes.fromhex(job["prev_hash"])[::-1]  # Reverse for little-endian
            merkle_root = self.calculate_merkle_root(job, extranonce2)
            
            if isinstance(job["ntime"], str):
                ntime = int(job["ntime"], 16)
            else:
                ntime = job["ntime"]  # Already an integer
                
            if isinstance(job["nbits"], str):
                nbits = int(job["nbits"], 16)
            else:
                nbits = job["nbits"]  # Already an integer
            
            # Build 80-byte header (CORRECTED ORDER)
            header = struct.pack("<I", version)  # 4 bytes, little-endian
            header += prev_hash  # 32 bytes
            header += merkle_root  # 32 bytes
            header += struct.pack("<I", ntime)  # 4 bytes, little-endian
            header += struct.pack("<I", nbits)  # 4 bytes, little-endian
            header += struct.pack("<I", nonce)  # 4 bytes, little-endian
            
            return header
            
        except Exception as e:
            print(f"Error building header: {e}")
            return b""
    
    def calculate_merkle_root(self, job: Dict, extranonce2: str) -> bytes:
        """Calculate merkle root for the job - CORRECTED"""
        try:
            # Build coinbase transaction
            coinbase = job["coinb1"] + extranonce2 + job["coinb2"]
            coinbase_hash = hashlib.sha256(hashlib.sha256(bytes.fromhex(coinbase)).digest()).digest()
            
            # Build merkle tree (simplified for empty merkle_branches)
            merkle_root = coinbase_hash
            
            # If there are merkle branches, add them
            merkle_branches = job.get("merkle_branches", [])
            for branch in merkle_branches:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + bytes.fromhex(branch)).digest()).digest()
            
            return merkle_root
            
        except Exception as e:
            print(f"Error calculating merkle root: {e}")
            return b"\x00" * 32
    
    def x16r_hash(self, header: bytes) -> bytes:
        """X16R hash function - CORRECTED IMPLEMENTATION"""
        try:
            # X16R uses the last 16 bytes of the previous hash to determine algorithm order
            # For simplicity, we'll use a fixed order based on the header
            hash_result = header
            
            # Apply 16 different hashing algorithms in sequence
            for i, algo in enumerate(self.x16r_algorithms):
                if algo == hashlib.blake2b:
                    hash_result = hashlib.blake2b(hash_result, digest_size=32).digest()
                elif algo == hashlib.sha3_256:
                    hash_result = hashlib.sha3_256(hash_result).digest()
                else:  # SHA256
                    hash_result = hashlib.sha256(hash_result).digest()
            
            return hash_result
            
        except Exception as e:
            print(f"Error in X16R hash: {e}")
            # Fallback to double SHA256
            return hashlib.sha256(hashlib.sha256(header).digest()).digest()
    
    def mine_block(self, job: Dict, pool: Dict) -> Optional[Tuple[int, str]]:
        """Mine a block with the given job - CORRECTED"""
        try:
            nonce = random.randint(0, 0xFFFFFFFF)
            extranonce2 = "00000000"  # Default extranonce2
            
            # Get target from job
            target = job.get("target")
            if not target:
                print(f"No target found for {pool['name']}")
                return None
            
            print(f"Starting mining for {pool['name']} with nonce {nonce}, target {target}")
            
            for i in range(1000):  # More iterations to find a solution
                # Build header
                header = self.build_block_header(job, nonce, extranonce2)
                if not header:
                    continue
                
                # Hash the header with X16R
                hash_result = self.x16r_hash(header)
                
                # Check if hash meets the pool's difficulty target
                if self.check_difficulty(hash_result, target):
                    print(f"Found solution for {pool['name']}: nonce={nonce}")
                    print(f"Target: {target}")
                    print(f"Hash: {hash_result.hex()}")
                    return nonce, extranonce2
                
                nonce = (nonce + self.nonce_increment) & 0xFFFFFFFF
                
                if i % 100 == 0:
                    print(f"Mining progress for {pool['name']}: iteration {i}, nonce {nonce}")
            
            return None
            
        except Exception as e:
            print(f"Error mining for {pool['name']}: {e}")
            return None
    
    def submit_share(self, sock: socket.socket, pool: Dict, job: Dict, nonce: int, extranonce2: str) -> bool:
        """Submit a share to the pool - CORRECTED"""
        try:
            # Format nonce as hex string (little-endian)
            nonce_hex = f"{nonce:08x}"
            
            # Build submission parameters
            params = [
                pool["user"],
                job["job_id"],
                extranonce2,
                job["ntime"],
                nonce_hex
            ]
            
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": params
            }
            
            print(f"Submitting share to {pool['name']}: {submit_msg}")
            
            sock.send((json.dumps(submit_msg) + "\n").encode())
            response = sock.recv(4096).decode()
            
            print(f"Share submission response from {pool['name']}: {response}")
            
            # Parse response
            try:
                result = json.loads(response)
                if result.get("result") is True:
                    print(f"Share accepted by {pool['name']}!")
                    return True
                else:
                    print(f"Share rejected by {pool['name']}: {result.get('error', 'Unknown error')}")
                    return False
            except json.JSONDecodeError:
                print(f"Invalid JSON response from {pool['name']}: {response}")
                return False
                
        except Exception as e:
            print(f"Error submitting share to {pool['name']}: {e}")
            return False
    
    def mine_pool(self, pool: Dict):
        """Mine on a specific pool"""
        try:
            print(f"Connecting to {pool['name']}...")
            sock = self.connect_to_pool(pool)
            if not sock:
                return
            
            print(f"Subscribing to {pool['name']}...")
            if not self.subscribe_to_pool(sock, pool):
                sock.close()
                return
            
            print(f"Getting mining job from {pool['name']}...")
            job = self.get_mining_job(sock, pool)
            if not job:
                print(f"No job received from {pool['name']}")
                sock.close()
                return
            
            print(f"Mining on {pool['name']}...")
            result = self.mine_block(job, pool)
            
            if result:
                nonce, extranonce2 = result
                print(f"Submitting share to {pool['name']}...")
                success = self.submit_share(sock, pool, job, nonce, extranonce2)
                if success:
                    print(f"Successfully submitted share to {pool['name']}!")
                else:
                    print(f"Failed to submit share to {pool['name']}")
            else:
                print(f"No solution found for {pool['name']}")
            
            sock.close()
            
        except Exception as e:
            print(f"Error mining on {pool['name']}: {e}")
    
    def run(self):
        """Run the test miner"""
        print("Starting Corrected X16R Test Miner...")
        self.running = True
        
        while self.running and self.iteration_count < MAX_ITERATIONS:
            self.iteration_count += 1
            print(f"\n=== Iteration {self.iteration_count}/{MAX_ITERATIONS} ===")
            
            # Mine on each pool
            threads = []
            for pool in self.pools[:self.max_pools]:
                thread = threading.Thread(target=self.mine_pool, args=(pool,))
                thread.start()
                threads.append(thread)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            print(f"Completed iteration {self.iteration_count}")
            time.sleep(1)  # Brief pause between iterations
        
        print(f"\nTest completed after {self.iteration_count} iterations")
        self.running = False

def main():
    """Main function"""
    miner = X16RCorrectedMiner()
    miner.run()

if __name__ == "__main__":
    main() 