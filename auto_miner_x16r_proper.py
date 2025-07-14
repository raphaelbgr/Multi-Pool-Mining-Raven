#!/usr/bin/env python3
"""
AUTO MINER X16R PROPER - Ravencoin Multi-Pool Miner with Proper X16R Algorithm
Based on official Ravencoin implementation and X16R algorithm
"""

import json
import subprocess
import time
import logging
import socket
import struct
import hashlib
import threading
from datetime import datetime
from typing import Dict, List, Optional

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner_x16r_proper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class X16RProperMiner:
    def __init__(self):
        self.config = self.load_config()
        self.nonce_start = 0
        self.cycle_count = 0
        self.total_shares = 0
        self.accepted_shares = 0
        self.x16r_miner = "miner_x16r_proper.exe"  # Proper X16R miner executable
        
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
            
            return target_hex
        except Exception as e:
            logger.error(f"Error deriving target from nbits: {e}")
            return "00000000ffff0000000000000000000000000000000000000000000000000000"
    
    def build_block_header(self, job: Dict, nonce: int, extranonce2: str = "") -> bytes:
        """Build Ravencoin block header for X16R"""
        try:
            logger.info(f"Building header for job {job.get('job_id', 'unknown')}")
            
            # Convert hex strings to bytes
            if isinstance(job["version"], str):
                version = int(job["version"], 16)
            else:
                version = job["version"]
            
            logger.info(f"Version: {version}")
            
            prev_hash = bytes.fromhex(job["prev_hash"])[::-1]  # Reverse for little-endian
            logger.info(f"Prev hash: {job['prev_hash']} -> {len(prev_hash)} bytes")
            
            merkle_root = self.calculate_merkle_root(job, extranonce2)
            logger.info(f"Merkle root: {len(merkle_root)} bytes")
            
            if isinstance(job["ntime"], str):
                ntime = int(job["ntime"], 16)
            else:
                ntime = job["ntime"]
            
            logger.info(f"Ntime: {ntime}")
            
            if isinstance(job["nbits"], str):
                nbits = int(job["nbits"], 16)
            else:
                nbits = job["nbits"]
            
            logger.info(f"Nbits: {nbits}")
            
            # Build 80-byte header
            header = struct.pack('<I', version)  # Version (4 bytes, little-endian)
            header += prev_hash  # Previous block hash (32 bytes)
            header += merkle_root  # Merkle root (32 bytes)
            header += struct.pack('<I', ntime)  # Timestamp (4 bytes, little-endian)
            header += struct.pack('<I', nbits)  # Bits (4 bytes, little-endian)
            header += struct.pack('<I', nonce)  # Nonce (4 bytes, little-endian)
            
            logger.info(f"Header built successfully: {len(header)} bytes")
            return header
            
        except Exception as e:
            logger.error(f"Error building header: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            valid_jobs = 0
            with open("headers.bin", "wb") as f:
                for i, job in enumerate(jobs):
                    if not job:
                        logger.warning(f"Job {i} is None, skipping")
                        continue
                    
                    logger.info(f"Processing job {i} for {job.get('job_id', 'unknown')}")
                    
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
                        valid_jobs += 1
                        logger.info(f"Successfully wrote header for job {i}")
                    else:
                        logger.error(f"Failed to build header for job {i}")
            
            logger.info(f"Created headers.bin with {valid_jobs} valid jobs out of {len(jobs)} total")
            return valid_jobs > 0
            
        except Exception as e:
            logger.error(f"Error creating headers.bin: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def check_x16r_miner(self):
        """Check if X16R miner is available"""
        try:
            result = subprocess.run([f"./{self.x16r_miner}", "0"], 
                                  capture_output=True, text=True, timeout=5)
            return True
        except:
            return False
    
    def run_x16r_mining_cycle(self, nonce_start: int, jobs: List[Optional[Dict]]) -> List[Optional[int]]:
        """Run one X16R mining cycle"""
        try:
            # Create headers.bin with real jobs
            self.create_headers_bin(jobs)
            
            # Run X16R miner
            cmd = [f"./{self.x16r_miner}", str(nonce_start)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("X16R mining cycle completed")
                # Parse results (simplified)
                return [None] * len(self.config['pools'])
            else:
                logger.error(f"X16R mining failed: {result.stderr}")
                return [None] * len(self.config['pools'])
                
        except Exception as e:
            logger.error(f"Error in X16R mining cycle: {e}")
            return [None] * len(self.config['pools'])
    
    def submit_share(self, pool: Dict, job: Dict, nonce: int, extranonce2: str) -> bool:
        """Submit share to pool"""
        try:
            sock = self.connect_to_pool(pool)
            if not sock:
                return False
            
            # Submit share
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": [
                    pool['user'],
                    job['job_id'],
                    extranonce2,
                    job['ntime'],
                    hex(nonce)[2:]
                ]
            }
            
            sock.sendall((json.dumps(submit_msg) + "\n").encode())
            response = sock.recv(4096).decode()
            
            logger.info(f"Share submission response from {pool['name']}: {response}")
            
            if '"result":true' in response or '"error":null' in response:
                self.accepted_shares += 1
                logger.info(f"Share accepted by {pool['name']}!")
                return True
            else:
                logger.warning(f"Share rejected by {pool['name']}: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting share to {pool['name']}: {e}")
            return False
    
    def run_cycle(self):
        """Run one mining cycle"""
        self.cycle_count += 1
        logger.info(f"=== Mining Cycle #{self.cycle_count} ===")
        
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
                            jobs.append(None)
                    else:
                        jobs.append(None)
                    sock.close()
                else:
                    jobs.append(None)
            except Exception as e:
                logger.error(f"Error with {pool['name']}: {e}")
                jobs.append(None)
        
        # Run X16R mining
        results = self.run_x16r_mining_cycle(self.nonce_start, jobs)
        
        # Submit shares
        for i, (pool, job, nonce) in enumerate(zip(self.config['pools'], jobs, results)):
            if job and nonce:
                self.submit_share(pool, job, nonce, "00000000")
                self.total_shares += 1
        
        # Update nonce start
        self.nonce_start += 4194304  # 4M nonces per cycle
        
        # Log statistics
        logger.info(f"Cycle completed. Total shares: {self.total_shares}, Accepted: {self.accepted_shares}")
    
    def start_mining(self):
        """Start X16R mining"""
        logger.info("Starting Proper X16R Ravencoin Miner")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("Using official X16R algorithm for Ravencoin")
        
        # Check if X16R miner exists
        if not self.check_x16r_miner():
            logger.error(f"X16R miner not found: {self.x16r_miner}")
            logger.error("Please compile the X16R miner first:")
            logger.error("  build_x16r_proper.bat")
            return
        
        logger.info("X16R miner found and ready")
        
        while True:
            try:
                self.run_cycle()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping X16R miner...")
                break
            except Exception as e:
                logger.error(f"Critical error: {str(e)}")
                time.sleep(5)

def main():
    """Main function"""
    miner = X16RProperMiner()
    miner.start_mining()

if __name__ == "__main__":
    main() 