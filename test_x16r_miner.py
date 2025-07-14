#!/usr/bin/env python3
"""
Test X16R Miner - Limited Iterations
Runs for a specific number of iterations then exits to check results
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
MAX_ITERATIONS = 10  # Run for 10 iterations then exit
NONCE_INCREMENT = 1000000
JOB_TIMEOUT = 30

class X16RTestMiner:
    def __init__(self, config_file: str = "config.json"):
        self.config = self.load_config(config_file)
        self.pools = self.config.get("pools", [])
        self.max_pools = self.config.get("max_pools", 32)
        self.nonce_increment = self.config.get("nonce_increment", NONCE_INCREMENT)
        
        self.current_jobs = {}
        self.pool_connections = {}
        self.running = False
        self.iteration_count = 0
        
        print(f"X16R Test Miner initialized with {len(self.pools)} pools")
        print(f"Will run for {MAX_ITERATIONS} iterations then exit")
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
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
            sock.settimeout(None)
            return sock
        except Exception as e:
            print(f"Failed to connect to {pool['name']}: {e}")
            return None
    
    def subscribe_to_pool(self, sock: socket.socket, pool: Dict) -> bool:
        """Subscribe to mining notifications"""
        try:
            # Stratum subscription
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": ["X16R-Miner/1.0.0"]
            }
            
            sock.send((json.dumps(subscribe_msg) + "\n").encode())
            response = sock.recv(4096).decode()
            
            print(f"Subscription response: {response}")
            
            # Authorize
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [pool["user"], pool["password"]]
            }
            
            sock.send((json.dumps(auth_msg) + "\n").encode())
            response = sock.recv(4096).decode()
            
            print(f"Authorization response: {response}")
            
            return True
        except Exception as e:
            print(f"Failed to subscribe to {pool['name']}: {e}")
            return False
    
    def get_mining_job(self, sock: socket.socket, pool: Dict) -> Optional[Dict]:
        """Get mining job from pool"""
        try:
            # Wait for mining.notify
            data = sock.recv(4096).decode()
            if not data:
                return None
            
            print(f"Received job data from {pool['name']}: {data}")
            
            # Parse the job notification
            lines = data.strip().split('\n')
            for line in lines:
                if not line:
                    continue
                
                try:
                    msg = json.loads(line)
                    if msg.get("method") == "mining.notify":
                        params = msg["params"]
                        # Handle alternate/shortened Ravencoin format (7 params, no merkle branches)
                        if len(params) == 7:
                            job_id = params[0]
                            prev_hash = params[1]
                            coinb1 = params[2]
                            coinb2 = params[3]
                            clean_jobs = params[4]  # This is actually the clean_jobs boolean
                            version = params[5]
                            nbits = params[6]
                            ntime = None  # Not provided, will set below
                            merkle_branches = []
                            # Try to get ntime from previous set_target or use current time
                            # For now, use current time as hex
                            import time
                            ntime = f"{int(time.time()):x}"
                            job = {
                                "job_id": job_id,
                                "prev_hash": prev_hash,
                                "coinb1": coinb1,
                                "coinb2": coinb2,
                                "merkle_branches": merkle_branches,
                                "version": f"{version:x}" if isinstance(version, int) else str(version),
                                "nbits": f"{nbits:x}" if isinstance(nbits, int) else str(nbits),
                                "ntime": ntime,
                                "clean_jobs": clean_jobs,
                                "pool": pool["name"]
                            }
                            print(f"Parsed job for {pool['name']}: {job}")
                            return job
                        # Fallback to previous logic for 8+ params
                        elif len(params) >= 8:
                            job_id = params[0]
                            prev_hash = params[1]
                            coinb1 = params[2]
                            coinb2 = params[3]
                            clean_jobs = params[4]
                            version = params[5]
                            nbits = params[6]
                            ntime = params[7]
                            merkle_branches = []
                            job = {
                                "job_id": job_id,
                                "prev_hash": prev_hash,
                                "coinb1": coinb1,
                                "coinb2": coinb2,
                                "merkle_branches": merkle_branches,
                                "version": f"{version:x}" if isinstance(version, int) else str(version),
                                "nbits": f"{nbits:x}" if isinstance(nbits, int) else str(nbits),
                                "ntime": ntime,
                                "clean_jobs": clean_jobs,
                                "pool": pool["name"]
                            }
                            print(f"Parsed job for {pool['name']}: {job}")
                            return job
                        else:
                            print(f"Invalid mining.notify parameters for {pool['name']}: {params}")
                        
                except json.JSONDecodeError:
                    continue
                except IndexError as e:
                    print(f"Index error parsing job from {pool['name']}: {e}")
                    continue
            
            return None
        except Exception as e:
            print(f"Error getting job from {pool['name']}: {e}")
            return None
    
    def build_block_header(self, job: Dict, nonce: int, extranonce2: str = "") -> bytes:
        """Build Ravencoin block header for X16R"""
        try:
            # Convert hex strings to bytes
            version = int(job["version"], 16)
            prev_hash = bytes.fromhex(job["prev_hash"])[::-1]  # Reverse for little-endian
            merkle_root = self.calculate_merkle_root(job, extranonce2)
            ntime = int(job["ntime"], 16)
            nbits = int(job["nbits"], 16)
            
            # Build 80-byte header
            header = struct.pack("<I", version)  # 4 bytes, little-endian
            header += prev_hash  # 32 bytes
            header += merkle_root  # 32 bytes
            header += struct.pack("<I", ntime)  # 4 bytes, little-endian
            header += struct.pack("<I", nbits)  # 4 bytes, little-endian
            header += struct.pack("<I", nonce)  # 4 bytes, little-endian
            
            print(f"Built header: version={version}, ntime={ntime}, nbits={nbits}, nonce={nonce}")
            return header
            
        except Exception as e:
            print(f"Error building header: {e}")
            return b""
    
    def calculate_merkle_root(self, job: Dict, extranonce2: str) -> bytes:
        """Calculate merkle root for the job"""
        try:
            # Build coinbase transaction
            coinbase = job["coinb1"] + extranonce2 + job["coinb2"]
            coinbase_hash = hashlib.sha256(hashlib.sha256(bytes.fromhex(coinbase)).digest()).digest()
            
            # Build merkle tree
            merkle_branches = job["merkle_branches"]
            merkle_root = coinbase_hash
            
            for branch in merkle_branches:
                merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + bytes.fromhex(branch)).digest()).digest()
            
            return merkle_root
            
        except Exception as e:
            print(f"Error calculating merkle root: {e}")
            return b"\x00" * 32
    
    def x16r_hash(self, header: bytes) -> bytes:
        """Simplified X16R hash function (placeholder)"""
        # This is a simplified version - real X16R uses 16 different algorithms
        # For testing, we'll use a simple double SHA256
        return hashlib.sha256(hashlib.sha256(header).digest()).digest()
    
    def mine_block(self, job: Dict, pool: Dict) -> Optional[Tuple[int, str]]:
        """Mine a block with the given job"""
        try:
            nonce = random.randint(0, 0xFFFFFFFF)
            extranonce2 = "00000000"  # Default extranonce2
            
            print(f"Starting mining for {pool['name']} with nonce {nonce}")
            
            for i in range(1000):  # Limited iterations for testing
                # Build header
                header = self.build_block_header(job, nonce, extranonce2)
                if not header:
                    continue
                
                # Print full header and submission details for debugging
                print(f"\n[DEBUG] Submission details for {pool['name']}:")
                print(f"  job_id: {job['job_id']}")
                print(f"  prev_hash: {job['prev_hash']}")
                print(f"  coinb1: {job['coinb1']}")
                print(f"  coinb2: {job['coinb2']}")
                print(f"  merkle_branches: {job['merkle_branches']}")
                print(f"  version: {job['version']}")
                print(f"  nbits: {job['nbits']}")
                print(f"  ntime: {job['ntime']}")
                print(f"  nonce: {nonce} (hex: {nonce:08x})")
                print(f"  extranonce2: {extranonce2}")
                print(f"  header (hex): {header.hex()}")
                
                # Hash the header
                hash_result = self.x16r_hash(header)
                print(f"  hash_result: {hash_result.hex()}")
                
                # Check if hash meets difficulty (simplified)
                if hash_result[0] == 0:  # Very low difficulty for testing
                    print(f"Found potential solution for {pool['name']}: nonce={nonce}")
                    return nonce, extranonce2
                
                nonce = (nonce + self.nonce_increment) & 0xFFFFFFFF
                
                if i % 100 == 0:
                    print(f"Mining progress for {pool['name']}: iteration {i}, nonce {nonce}")
            
            return None
            
        except Exception as e:
            print(f"Error mining for {pool['name']}: {e}")
            return None
    
    def submit_share(self, sock: socket.socket, pool: Dict, job: Dict, nonce: int, extranonce2: str) -> bool:
        """Submit a share to the pool"""
        try:
            # Format nonce as hex string
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
        print("Starting X16R Test Miner...")
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
    miner = X16RTestMiner()
    miner.run()

if __name__ == "__main__":
    main() 