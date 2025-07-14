#!/usr/bin/env python3
"""
TEST X16R PROPER MINER - Limited Test of Proper X16R Implementation
Tests the proper X16R miner with real pool connections and limited cycles
"""

import json
import subprocess
import time
import logging
import socket
import struct
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_x16r_proper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class X16RProperTester:
    def __init__(self):
        self.config = self.load_config()
        self.nonce_start = 0
        self.cycle_count = 0
        self.total_shares = 0
        self.accepted_shares = 0
        self.x16r_miner = "miner_x16r_proper.exe"
        self.max_cycles = 3  # Run for 3 cycles only
        
    def load_config(self):
        try:
            with open("config.json") as f:
                config = json.load(f)
            logger.info("Config loaded successfully")
            logger.info(f"Configured pools: {len(config['pools'])}")
            for pool in config['pools']:
                logger.info(f"  - {pool['name']}: {pool['host']}:{pool['port']}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise
    
    def connect_to_pool(self, pool: Dict) -> Optional[socket.socket]:
        """Connect to a mining pool"""
        try:
            sock = socket.create_connection((pool['host'], pool['port']), timeout=10)
            logger.info(f"Connected to {pool['name']}")
            return sock
        except Exception as e:
            logger.error(f"Failed to connect to {pool['name']}: {e}")
            return None
    
    def subscribe_to_pool(self, sock: socket.socket, pool: Dict) -> bool:
        """Subscribe to mining pool"""
        try:
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            sock.sendall((json.dumps(subscribe_msg) + "\n").encode())
            
            response = sock.recv(4096).decode()
            logger.info(f"Subscribe response from {pool['name']}: {response}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {pool['name']}: {e}")
            return False
    
    def authenticate_to_pool(self, sock: socket.socket, pool: Dict) -> bool:
        """Authenticate with mining pool"""
        try:
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [pool['user'], pool['password']]
            }
            sock.sendall((json.dumps(auth_msg) + "\n").encode())
            
            response = sock.recv(4096).decode()
            logger.info(f"Auth response from {pool['name']}: {response}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate to {pool['name']}: {e}")
            return False
    
    def get_mining_job(self, sock: socket.socket, pool: Dict) -> Optional[Dict]:
        """Get mining job from pool"""
        try:
            data = sock.recv(4096).decode()
            if not data:
                return None
            
            lines = data.strip().split('\n')
            job = None
            target = None
            
            for line in lines:
                if not line:
                    continue
                
                try:
                    msg = json.loads(line)
                    
                    if msg.get("method") == "mining.notify":
                        params = msg["params"]
                        if len(params) >= 7:
                            job = {
                                'job_id': params[0],
                                'prev_hash': params[1],
                                'coinb1': params[2],
                                'coinb2': params[3],
                                'merkle_branches': [],
                                'version': params[5],
                                'nbits': params[6],
                                'ntime': None,
                                'clean_jobs': params[4] if len(params) > 4 else True,
                                'target': None
                            }
                            
                            # Generate ntime if not provided
                            if len(params) > 7:
                                job['ntime'] = params[7]
                            else:
                                job['ntime'] = hex(int(time.time()))[2:]
                    
                    elif msg.get("method") == "mining.set_target":
                        target = msg["params"][0]
                        if job:
                            job['target'] = target
                
                except json.JSONDecodeError:
                    continue
            
            if job:
                if not job['target']:
                    # Derive target from nbits
                    job['target'] = self.derive_target_from_nbits(job['nbits'])
                
                logger.info(f"Parsed job for {pool['name']}: {job}")
                return job
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting job from {pool['name']}: {e}")
            return None
    
    def derive_target_from_nbits(self, nbits: str) -> str:
        """Derive target from nbits"""
        try:
            if isinstance(nbits, int):
                nbits_hex = f"{nbits:08x}"
            else:
                nbits_hex = nbits.zfill(8)
            
            exponent = int(nbits_hex[0:2], 16)
            mantissa = int(nbits_hex[2:], 16)
            
            target_int = mantissa * (1 << (8 * (exponent - 3)))
            target_bytes = target_int.to_bytes(32, byteorder='big')
            target_hex = target_bytes.hex()
            
            logger.info(f"Derived target from nbits {nbits}: {target_hex}")
            return target_hex
        except Exception as e:
            logger.error(f"Error deriving target from nbits: {e}")
            return "00000000ffff0000000000000000000000000000000000000000000000000000"
    
    def build_block_header(self, job: Dict, nonce: int, extranonce2: str = "") -> bytes:
        """Build Ravencoin block header for X16R"""
        try:
            # Convert hex strings to bytes
            if isinstance(job["version"], str):
                version = int(job["version"], 16)
            else:
                version = job["version"]
            
            prev_hash = bytes.fromhex(job["prev_hash"])[::-1]  # Reverse for little-endian
            merkle_root = self.calculate_merkle_root(job, extranonce2)
            
            if isinstance(job["ntime"], str):
                ntime = int(job["ntime"], 16)
            else:
                ntime = job["ntime"]
            
            if isinstance(job["nbits"], str):
                nbits = int(job["nbits"], 16)
            else:
                nbits = job["nbits"]
            
            # Build 80-byte header
            header = struct.pack('<I', version)  # Version (4 bytes, little-endian)
            header += prev_hash  # Previous block hash (32 bytes)
            header += merkle_root  # Merkle root (32 bytes)
            header += struct.pack('<I', ntime)  # Timestamp (4 bytes, little-endian)
            header += struct.pack('<I', nbits)  # Bits (4 bytes, little-endian)
            header += struct.pack('<I', nonce)  # Nonce (4 bytes, little-endian)
            
            return header
            
        except Exception as e:
            logger.error(f"Error building header: {e}")
            return None
    
    def calculate_merkle_root(self, job: Dict, extranonce2: str) -> bytes:
        """Calculate merkle root"""
        try:
            # Build coinbase transaction
            coinbase = job["coinb1"] + extranonce2 + job["coinb2"]
            coinbase_hash = hashlib.sha256(hashlib.sha256(bytes.fromhex(coinbase)).digest()).digest()
            
            # For simplicity, use coinbase hash as merkle root
            # In production, you'd build the full merkle tree
            return coinbase_hash
            
        except Exception as e:
            logger.error(f"Error calculating merkle root: {e}")
            return bytes(32)
    
    def create_headers_bin(self, jobs: List[Dict]) -> bool:
        """Create headers.bin file for CUDA miner"""
        try:
            with open("headers.bin", "wb") as f:
                for i, job in enumerate(jobs):
                    if not job:
                        logger.warning(f"No job for pool {i}, skipping")
                        continue
                    
                    # Create header with real job data
                    header = self.build_block_header(job, 0, "00000000")
                    if header:
                        # Write header length (4 bytes)
                        f.write(struct.pack('<I', len(header)))
                        # Write header (80 bytes)
                        f.write(header)
                        # Write target (32 bytes)
                        target_bytes = bytes.fromhex(job.get('target', '00000000ffff0000000000000000000000000000000000000000000000000000'))
                        f.write(target_bytes)
                        logger.info(f"Added job {i} to headers.bin: {job['job_id']}")
                    else:
                        logger.error(f"Failed to build header for job {i}")
            
            logger.info(f"Created headers.bin with {len([j for j in jobs if j])} valid jobs")
            return True
            
        except Exception as e:
            logger.error(f"Error creating headers.bin: {e}")
            return False
    
    def check_x16r_miner(self):
        """Check if X16R miner is available"""
        try:
            result = subprocess.run([f"./{self.x16r_miner}", "0"], 
                                  capture_output=True, text=True, timeout=5)
            return True
        except:
            return False
    
    def run_x16r_mining_cycle(self, nonce_start: int) -> List[Optional[int]]:
        """Run one X16R mining cycle"""
        try:
            logger.info(f"Running X16R mining cycle with nonce_start: {nonce_start}")
            
            # Run X16R miner
            cmd = [f"./{self.x16r_miner}", str(nonce_start)]
            logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            logger.info(f"X16R miner stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"X16R miner stderr: {result.stderr}")
            
            if result.returncode == 0:
                logger.info("X16R mining cycle completed successfully")
                # Parse results (simplified - in real implementation you'd parse the output)
                return [None] * len(self.config['pools'])
            else:
                logger.error(f"X16R mining failed with return code: {result.returncode}")
                return [None] * len(self.config['pools'])
                
        except Exception as e:
            logger.error(f"Error in X16R mining cycle: {e}")
            return [None] * len(self.config['pools'])
    
    def run_cycle(self):
        """Run one mining cycle"""
        self.cycle_count += 1
        logger.info(f"=== Mining Cycle #{self.cycle_count}/{self.max_cycles} ===")
        
        # Get jobs from all pools
        jobs = []
        for pool in self.config['pools']:
            try:
                sock = self.connect_to_pool(pool)
                if sock:
                    if self.subscribe_to_pool(sock, pool):
                        if self.authenticate_to_pool(sock, pool):
                            job = self.get_mining_job(sock, pool)
                            jobs.append(job)
                        else:
                            logger.warning(f"Authentication failed for {pool['name']}")
                            jobs.append(None)
                    else:
                        logger.warning(f"Subscription failed for {pool['name']}")
                        jobs.append(None)
                    sock.close()
                else:
                    logger.warning(f"Connection failed for {pool['name']}")
                    jobs.append(None)
            except Exception as e:
                logger.error(f"Error with {pool['name']}: {e}")
                jobs.append(None)
        
        # Create headers.bin with real jobs
        if self.create_headers_bin(jobs):
            # Run X16R mining
            results = self.run_x16r_mining_cycle(self.nonce_start)
            
            # Log results
            logger.info(f"X16R mining results: {results}")
            
            # Update nonce start
            self.nonce_start += 4194304  # 4M nonces per cycle
            
            # Log statistics
            logger.info(f"Cycle completed. Total shares: {self.total_shares}, Accepted: {self.accepted_shares}")
        else:
            logger.error("Failed to create headers.bin, skipping mining cycle")
    
    def start_test(self):
        """Start X16R test mining"""
        logger.info("Starting Proper X16R Test Miner")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("Using official X16R algorithm for Ravencoin")
        logger.info(f"Will run for {self.max_cycles} cycles")
        
        # Check if X16R miner exists
        if not self.check_x16r_miner():
            logger.error(f"X16R miner not found: {self.x16r_miner}")
            logger.error("Please compile the X16R miner first:")
            logger.error("  build_x16r_proper.bat")
            return
        
        logger.info("X16R miner found and ready")
        
        # Run limited cycles
        for cycle in range(self.max_cycles):
            try:
                self.run_cycle()
                if cycle < self.max_cycles - 1:  # Don't sleep after last cycle
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Critical error in cycle {cycle + 1}: {str(e)}")
                break
        
        logger.info("Test completed!")
        logger.info(f"Final statistics: Total shares: {self.total_shares}, Accepted: {self.accepted_shares}")

def main():
    """Main function"""
    tester = X16RProperTester()
    tester.start_test()

if __name__ == "__main__":
    main() 