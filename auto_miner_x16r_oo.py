#!/usr/bin/env python3
"""
OBJECT-ORIENTED X16R MINER - Ravencoin Multi-Pool Miner
Each pool has its own header object and CUDA miner instance
"""

import json
import subprocess
import time
import logging
import socket
import struct
import hashlib
import threading
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner_x16r_oo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MiningJob:
    """Represents a mining job from a pool"""
    job_id: str
    prev_hash: str
    coinb1: str
    coinb2: str
    version: int
    nbits: str
    ntime: str
    target: str
    clean_jobs: bool = True
    merkle_branches: List[str] = None
    
    def __post_init__(self):
        if self.merkle_branches is None:
            self.merkle_branches = []

class PoolHeader:
    """Represents a pool's block header for mining"""
    
    def __init__(self, pool_name: str, job: MiningJob):
        self.pool_name = pool_name
        self.job = job
        self.header_file = f"header_{pool_name.lower()}.bin"
        self.nonce_start = 0
        self.last_solution = None
        
    def build_header(self, nonce: int, extranonce2: str = "") -> bytes:
        """Build Ravencoin block header for X16R"""
        try:
            # Convert hex strings to bytes
            version = self.job.version
            prev_hash = bytes.fromhex(self.job.prev_hash)[::-1]  # Reverse for little-endian
            merkle_root = self.calculate_merkle_root(extranonce2)
            
            if isinstance(self.job.ntime, str):
                ntime = int(self.job.ntime, 16)
            else:
                ntime = self.job.ntime
            
            if isinstance(self.job.nbits, str):
                nbits = int(self.job.nbits, 16)
            else:
                nbits = self.job.nbits
            
            # Build 80-byte header
            header = struct.pack('<I', version)  # Version (4 bytes, little-endian)
            header += prev_hash  # Previous block hash (32 bytes)
            header += merkle_root  # Merkle root (32 bytes)
            header += struct.pack('<I', ntime)  # Timestamp (4 bytes, little-endian)
            header += struct.pack('<I', nbits)  # Bits (4 bytes, little-endian)
            header += struct.pack('<I', nonce)  # Nonce (4 bytes, little-endian)
            
            return header
            
        except Exception as e:
            logger.error(f"Error building header for {self.pool_name}: {e}")
            return None
    
    def calculate_merkle_root(self, extranonce2: str) -> bytes:
        """Calculate merkle root"""
        try:
            # Build coinbase transaction
            coinbase = self.job.coinb1 + extranonce2 + self.job.coinb2
            coinbase_hash = hashlib.sha256(hashlib.sha256(bytes.fromhex(coinbase)).digest()).digest()
            
            # For simplicity, use coinbase hash as merkle root
            # In production, you'd build the full merkle tree
            return coinbase_hash
            
        except Exception as e:
            logger.error(f"Error calculating merkle root for {self.pool_name}: {e}")
            return bytes(32)
    
    def create_header_file(self) -> bool:
        """Create header file for CUDA miner"""
        try:
            with open(self.header_file, "wb") as f:
                # Create header
                header = self.build_header(0, "00000000")
                if header:
                    # Write header length (4 bytes)
                    f.write(struct.pack('<I', len(header)))
                    # Write header (80 bytes)
                    f.write(header)
                    # Write target (32 bytes)
                    target_bytes = bytes.fromhex(self.job.target)
                    f.write(target_bytes)
                    
                    logger.info(f"Created {self.header_file} for {self.pool_name}")
                    return True
                else:
                    logger.error(f"Failed to build header for {self.pool_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating {self.header_file}: {e}")
            return False
    
    def cleanup(self):
        """Clean up header file"""
        if os.path.exists(self.header_file):
            try:
                os.remove(self.header_file)
            except:
                pass

class PoolConnection:
    """Represents a connection to a mining pool"""
    
    def __init__(self, pool_config: Dict):
        self.name = pool_config['name']
        self.host = pool_config['host']
        self.port = pool_config['port']
        self.user = pool_config['user']
        self.password = pool_config['password']
        self.adapter = pool_config.get('adapter', 'stratum1')
        self.extra = pool_config.get('extra', {})
        self.socket = None
        self.current_job = None
        self.header = None
        
    def connect(self) -> bool:
        """Connect to the mining pool"""
        try:
            self.socket = socket.create_connection((self.host, self.port), timeout=10)
            logger.info(f"Connected to {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    def subscribe(self) -> bool:
        """Subscribe to mining pool"""
        try:
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            self.socket.sendall((json.dumps(subscribe_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Subscribe response from {self.name}: {response}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {self.name}: {e}")
            return False
    
    def authenticate(self) -> bool:
        """Authenticate with mining pool"""
        try:
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [self.user, self.password]
            }
            self.socket.sendall((json.dumps(auth_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Auth response from {self.name}: {response}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate to {self.name}: {e}")
            return False
    
    def get_job(self) -> Optional[MiningJob]:
        """Get mining job from pool"""
        try:
            data = self.socket.recv(4096).decode()
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
                            job = MiningJob(
                                job_id=params[0],
                                prev_hash=params[1],
                                coinb1=params[2],
                                coinb2=params[3],
                                version=params[5],
                                nbits=params[6],
                                ntime=None,
                                target=None,
                                clean_jobs=params[4] if len(params) > 4 else True
                            )
                            
                            # Generate ntime if not provided
                            if len(params) > 7:
                                job.ntime = params[7]
                            else:
                                job.ntime = hex(int(time.time()))[2:]
                    
                    elif msg.get("method") == "mining.set_target":
                        target = msg["params"][0]
                        if job:
                            job.target = target
                
                except json.JSONDecodeError:
                    continue
            
            if job:
                if not job.target:
                    # Derive target from nbits
                    job.target = self.derive_target_from_nbits(job.nbits)
                
                logger.info(f"Parsed job for {self.name}: {job}")
                return job
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting job from {self.name}: {e}")
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
    
    def submit_share(self, nonce: int, extranonce2: str = "00000000") -> bool:
        """Submit share to pool"""
        try:
            if not self.current_job:
                logger.error(f"No current job for {self.name}")
                return False
            
            # Submit share
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": [
                    self.user,
                    self.current_job.job_id,
                    extranonce2,
                    self.current_job.ntime,
                    hex(nonce)[2:]
                ]
            }
            
            self.socket.sendall((json.dumps(submit_msg) + "\n").encode())
            response = self.socket.recv(4096).decode()
            
            logger.info(f"Share submission response from {self.name}: {response}")
            
            if '"result":true' in response or '"error":null' in response:
                logger.info(f"Share accepted by {self.name}!")
                return True
            else:
                logger.warning(f"Share rejected by {self.name}: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting share to {self.name}: {e}")
            return False
    
    def close(self):
        """Close connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass

class X16RMiner:
    """CUDA X16R miner for a specific pool"""
    
    def __init__(self, pool_name: str, miner_exe: str = "miner_x16r_proper.exe"):
        self.pool_name = pool_name
        self.miner_exe = miner_exe
        self.nonce_start = 0
        
    def mine(self, header_file: str, nonce_start: int) -> Optional[int]:
        """Run X16R mining on CUDA"""
        try:
            if not os.path.exists(header_file):
                logger.error(f"Header file not found: {header_file}")
                return None
            
            # Run X16R miner
            cmd = [f"./{self.miner_exe}", str(nonce_start)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"X16R mining completed for {self.pool_name}")
                # Parse results (simplified - in real implementation you'd parse the output)
                # For now, return None (no solution found)
                return None
            else:
                logger.error(f"X16R mining failed for {self.pool_name}: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error in X16R mining for {self.pool_name}: {e}")
            return None

class X16ROOMiner:
    """Object-oriented X16R multi-pool miner"""
    
    def __init__(self):
        self.config = self.load_config()
        self.pools = []
        self.miners = []
        self.cycle_count = 0
        self.total_shares = 0
        self.accepted_shares = 0
        
        # Initialize pools and miners
        for pool_config in self.config['pools']:
            pool = PoolConnection(pool_config)
            miner = X16RMiner(pool.name)
            self.pools.append(pool)
            self.miners.append(miner)
    
    def load_config(self):
        """Load configuration"""
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
    
    def check_x16r_miner(self):
        """Check if X16R miner is available"""
        try:
            result = subprocess.run([f"./miner_x16r_proper.exe", "0"], 
                                  capture_output=True, text=True, timeout=5)
            return True
        except:
            return False
    
    def run_cycle(self):
        """Run one mining cycle"""
        self.cycle_count += 1
        logger.info(f"=== Mining Cycle #{self.cycle_count} ===")
        
        # Get jobs from all pools
        for i, pool in enumerate(self.pools):
            try:
                if pool.connect():
                    if pool.subscribe():
                        if pool.authenticate():
                            job = pool.get_job()
                            if job:
                                pool.current_job = job
                                pool.header = PoolHeader(pool.name, job)
                                
                                # Create header file
                                if pool.header.create_header_file():
                                    # Run CUDA mining
                                    nonce_start = self.miners[i].nonce_start
                                    solution = self.miners[i].mine(pool.header.header_file, nonce_start)
                                    
                                    if solution:
                                        # Submit share
                                        if pool.submit_share(solution):
                                            self.accepted_shares += 1
                                        self.total_shares += 1
                                    
                                    # Update nonce start
                                    self.miners[i].nonce_start += 4194304  # 4M nonces per cycle
                                
                                # Cleanup
                                pool.header.cleanup()
                        
                    pool.close()
                    
            except Exception as e:
                logger.error(f"Error with {pool.name}: {e}")
                pool.close()
        
        # Log statistics
        logger.info(f"Cycle completed. Total shares: {self.total_shares}, Accepted: {self.accepted_shares}")
    
    def start_mining(self):
        """Start X16R mining"""
        logger.info("Starting Object-Oriented X16R Ravencoin Miner")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("Using CUDA X16R algorithm with per-pool headers")
        
        # Check if X16R miner exists
        if not self.check_x16r_miner():
            logger.error("X16R miner not found: miner_x16r_proper.exe")
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
    miner = X16ROOMiner()
    miner.start_mining()

if __name__ == "__main__":
    main() 