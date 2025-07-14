#!/usr/bin/env python3
"""
OPTIMIZED X16R MINER - Ravencoin Multi-Pool Miner
Single CUDA hash generation → test against all pool targets simultaneously
Parallel share submission with dynamic challenge updates
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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner_x16r_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PoolChallenge:
    """Represents a pool's mining challenge"""
    pool_name: str
    job_id: str
    header: bytes  # 80-byte block header
    target: bytes  # 32-byte target
    extranonce2: str
    ntime: str
    version: int
    clean_jobs: bool

class OptimizedX16RMiner:
    def __init__(self):
        self.config = self.load_config()
        self.x16r_miner = "miner_x16r_proper.exe"
        self.nonce_start = 0
        self.challenges: List[PoolChallenge] = []
        self.challenges_lock = threading.Lock()
        self.running = True
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info("Optimized X16R Miner initialized")
        logger.info(f"Configured pools: {len(self.config['pools'])}")
        for pool in self.config['pools']:
            logger.info(f"  - {pool['name']}: {pool['host']}:{pool['port']}")

    def load_config(self) -> Dict:
        """Load configuration from config.json"""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            logger.info("Config loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def recv_json_lines(self, sock: socket.socket, timeout: float = 10.0) -> list:
        """Receive and parse all JSON objects from the socket buffer (handles multi-line responses)"""
        sock.settimeout(timeout)
        data = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b'\n' in chunk:
                    break
        except socket.timeout:
            pass
        except Exception as e:
            logger.error(f"Socket error: {e}")
            return []
        lines = data.decode(errors="ignore").splitlines()
        objs = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Failed to parse JSON line: {line} ({e})")
        return objs

    def connect_to_pool(self, pool: Dict, max_retries: int = 3) -> Optional[socket.socket]:
        """Connect to pool with retry logic (for WoolyPooly and others)"""
        for attempt in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((pool['host'], pool['port']))
                logger.info(f"Connected to {pool['name']}")
                return sock
            except Exception as e:
                logger.warning(f"Failed to connect to {pool['name']} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        logger.error(f"Could not connect to {pool['name']} after {max_retries} attempts")
        return None

    def subscribe_to_pool(self, sock: socket.socket, pool: Dict) -> bool:
        """Subscribe to mining notifications"""
        try:
            # Stratum subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": [f"X16R-Miner/1.0", None]
            }
            sock.send((json.dumps(subscribe_msg) + "\n").encode())
            
            response = sock.recv(4096).decode().strip()
            logger.info(f"Subscribe response from {pool['name']}: {response}")
            
            # Parse extranonce from response
            data = json.loads(response)
            if 'result' in data and data['result']:
                extranonce_info = data['result'][1] if len(data['result']) > 1 else "00000000"
                pool['extranonce'] = extranonce_info
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to subscribe to {pool['name']}: {e}")
            return False

    def authenticate_to_pool(self, sock: socket.socket, pool: Dict) -> bool:
        """Authenticate with the pool, robust to multi-line JSON"""
        try:
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [pool['user'], pool['password']]
            }
            sock.send((json.dumps(auth_msg) + "\n").encode())
            responses = self.recv_json_lines(sock)
            for data in responses:
                if isinstance(data, dict):
                    if data.get('result', False) is True:
                        return True
                    if data.get('method') == 'mining.set_target':
                        continue  # not auth
            return False
        except Exception as e:
            logger.error(f"Failed to authenticate to {pool['name']}: {e}")
            return False

    def get_job_from_pool(self, sock: socket.socket, pool: Dict) -> Optional[Dict]:
        """Get mining job from pool, robust to multi-line and multi-message responses"""
        try:
            responses = self.recv_json_lines(sock)
            for data in responses:
                # Stratum1/2: look for mining.notify or result
                if isinstance(data, dict):
                    if data.get('method') == 'mining.notify':
                        params = data['params']
                        job = {
                            'job_id': params[0],
                            'prev_hash': params[1],
                            'coinb1': params[2],
                            'coinb2': params[3],
                            'merkle_branches': params[4] if len(params) > 4 and isinstance(params[4], list) else [],
                            'version': params[5] if len(params) > 5 else 0,
                            'nbits': params[6] if len(params) > 6 else '',
                            'ntime': params[7] if len(params) > 7 else '',
                            'clean_jobs': params[8] if len(params) > 8 else True,
                            'target': pool.get('target', '00000000ffff0000000000000000000000000000000000000000000000000000')
                        }
                        return job
                    elif 'result' in data and isinstance(data['result'], list) and len(data['result']) > 0:
                        # Some pools send job in result
                        continue  # handled elsewhere
            return None
        except Exception as e:
            logger.error(f"Error getting job from {pool['name']}: {e}")
            return None

    def build_block_header(self, job: Dict, extranonce2: str = "") -> Optional[bytes]:
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
            
            # Build 80-byte header: version(4) + prev_hash(32) + merkle_root(32) + ntime(4) + nbits(4) + nonce(4)
            header = struct.pack('<I', version)  # Version
            header += prev_hash  # Previous hash
            header += merkle_root  # Merkle root
            header += struct.pack('<I', ntime)  # Timestamp
            header += struct.pack('<I', nbits)  # Bits
            header += struct.pack('<I', 0)  # Nonce (placeholder)
            
            return header
        except Exception as e:
            logger.error(f"Failed to build header: {e}")
            return None

    def calculate_merkle_root(self, job: Dict, extranonce2: str) -> bytes:
        """Calculate merkle root for the job"""
        try:
            # Combine coinb1 + extranonce2 + coinb2
            coinbase = job["coinb1"] + extranonce2 + job["coinb2"]
            coinbase_hash = hashlib.sha256(hashlib.sha256(bytes.fromhex(coinbase)).digest()).digest()
            
            # Defensive: ensure merkle_branches is a list
            merkle_branches = job.get("merkle_branches")
            if not isinstance(merkle_branches, list):
                merkle_branches = []
            
            if merkle_branches:
                merkle_root = coinbase_hash
                for branch in merkle_branches:
                    merkle_root = hashlib.sha256(
                        hashlib.sha256(merkle_root + bytes.fromhex(branch)).digest()
                    ).digest()
                return merkle_root
            else:
                return coinbase_hash
        except Exception as e:
            logger.error(f"Failed to calculate merkle root: {e}")
            return bytes(32)

    def create_challenges_file(self) -> bool:
        """Create challenges.bin file with all current challenges"""
        try:
            with self.challenges_lock:
                if not self.challenges:
                    logger.warning("No challenges available")
                    return False
                
                with open("challenges.bin", "wb") as f:
                    for challenge in self.challenges:
                        # Write header length (4 bytes)
                        f.write(struct.pack('<I', len(challenge.header)))
                        # Write header (80 bytes)
                        f.write(challenge.header)
                        # Write target (32 bytes)
                        f.write(challenge.target)
                        # Write pool name length and name
                        pool_name_bytes = challenge.pool_name.encode()
                        f.write(struct.pack('<I', len(pool_name_bytes)))
                        f.write(pool_name_bytes)
                
                logger.info(f"Created challenges.bin with {len(self.challenges)} challenges")
                return True
        except Exception as e:
            logger.error(f"Failed to create challenges file: {e}")
            return False

    def run_cuda_mining_cycle(self, nonce_start: int) -> List[Tuple[str, int, str]]:
        """Run one CUDA mining cycle and return valid shares"""
        try:
            # Create challenges file
            if not self.create_challenges_file():
                return []
            
            # Run CUDA miner
            cmd = [f"./{self.x16r_miner}", str(nonce_start)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("CUDA mining cycle completed")
                # Parse results for valid shares
                shares = self.parse_mining_results(result.stdout)
                return shares
            else:
                logger.error(f"CUDA mining failed: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error in CUDA mining cycle: {e}")
            return []

    def parse_mining_results(self, output: str) -> List[Tuple[str, int, str]]:
        """Parse CUDA miner output for valid shares"""
        shares = []
        try:
            lines = output.strip().split('\n')
            for line in lines:
                if 'SHARE' in line or 'SOLUTION' in line:
                    # Parse share format: pool_name nonce extranonce2
                    parts = line.split()
                    if len(parts) >= 3:
                        pool_name = parts[0]
                        nonce = int(parts[1])
                        extranonce2 = parts[2]
                        shares.append((pool_name, nonce, extranonce2))
                        logger.info(f"Found valid share for {pool_name}: nonce={nonce}")
        except Exception as e:
            logger.error(f"Error parsing mining results: {e}")
        return shares

    def submit_share_parallel(self, pool_name: str, job_id: str, nonce: int, extranonce2: str, ntime: str) -> bool:
        """Submit share to pool in parallel"""
        try:
            # Find the pool configuration
            pool_config = None
            for pool in self.config['pools']:
                if pool['name'] == pool_name:
                    pool_config = pool
                    break
            
            if not pool_config:
                logger.error(f"Pool {pool_name} not found in config")
                return False
            
            # Connect and submit share
            sock = self.connect_to_pool(pool_config)
            if not sock:
                return False
            
            try:
                # Submit share
                submit_msg = {
                    "id": 4,
                    "method": "mining.submit",
                    "params": [pool_config['user'], job_id, extranonce2, ntime, hex(nonce)[2:]]
                }
                sock.send((json.dumps(submit_msg) + "\n").encode())
                
                response = sock.recv(4096).decode().strip()
                data = json.loads(response)
                
                if data.get('result', False) == True:
                    logger.info(f"Share accepted by {pool_name}")
                    return True
                else:
                    logger.warning(f"Share rejected by {pool_name}: {response}")
                    return False
                    
            finally:
                sock.close()
                
        except Exception as e:
            logger.error(f"Error submitting share to {pool_name}: {e}")
            return False

    def update_challenges(self):
        """Update challenges from all pools"""
        new_challenges = []
        
        for pool in self.config['pools']:
            try:
                # Connect to pool
                sock = self.connect_to_pool(pool)
                if not sock:
                    continue
                
                try:
                    # Subscribe and authenticate
                    if not self.subscribe_to_pool(sock, pool):
                        continue
                    
                    if not self.authenticate_to_pool(sock, pool):
                        continue
                    
                    # Get job
                    job = self.get_job_from_pool(sock, pool)
                    if not job:
                        continue
                    
                    # Build header
                    extranonce2 = pool.get('extranonce', '00000000')
                    header = self.build_block_header(job, extranonce2)
                    if not header:
                        continue
                    
                    # Create challenge
                    target_bytes = bytes.fromhex(job.get('target', '00000000ffff0000000000000000000000000000000000000000000000000000'))
                    challenge = PoolChallenge(
                        pool_name=pool['name'],
                        job_id=job['job_id'],
                        header=header,
                        target=target_bytes,
                        extranonce2=extranonce2,
                        ntime=job['ntime'],
                        version=job['version'],
                        clean_jobs=job['clean_jobs']
                    )
                    new_challenges.append(challenge)
                    logger.info(f"Created challenge for {pool['name']}")
                    
                finally:
                    sock.close()
                    
            except Exception as e:
                logger.error(f"Error updating challenge for {pool['name']}: {e}")
        
        # Update challenges list
        with self.challenges_lock:
            self.challenges = new_challenges

    def run_cycle(self):
        """Run one complete mining cycle"""
        cycle_start = time.time()
        logger.info(f"=== Mining Cycle #{self.nonce_start // self.config['nonce_increment'] + 1} ===")
        
        # Update challenges from all pools
        self.update_challenges()
        
        if not self.challenges:
            logger.warning("No challenges available, skipping cycle")
            return
        
        # Run CUDA mining
        shares = self.run_cuda_mining_cycle(self.nonce_start)
        
        # Submit shares in parallel
        if shares:
            futures = []
            for pool_name, nonce, extranonce2 in shares:
                # Find the challenge for this pool
                challenge = None
                for c in self.challenges:
                    if c.pool_name == pool_name:
                        challenge = c
                        break
                
                if challenge:
                    future = self.executor.submit(
                        self.submit_share_parallel,
                        pool_name,
                        challenge.job_id,
                        nonce,
                        extranonce2,
                        challenge.ntime
                    )
                    futures.append(future)
            
            # Wait for all submissions to complete
            accepted = 0
            for future in futures:
                if future.result():
                    accepted += 1
            
            logger.info(f"Cycle completed. Total shares: {len(shares)}, Accepted: {accepted}")
        else:
            logger.info("Cycle completed. No shares found")
        
        # Update nonce start for next cycle
        self.nonce_start += self.config['nonce_increment']

    def run(self):
        """Main mining loop"""
        logger.info("Starting Optimized X16R Ravencoin Miner")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("Using single CUDA hash generation with multi-pool target testing")
        
        # Check if CUDA miner exists
        if not os.path.exists(self.x16r_miner):
            logger.error(f"CUDA miner not found: {self.x16r_miner}")
            return
        
        logger.info("CUDA miner found and ready")
        
        try:
            while self.running:
                self.run_cycle()
                time.sleep(1)  # Brief pause between cycles
                
        except KeyboardInterrupt:
            logger.info("Stopping optimized X16R miner...")
        finally:
            self.executor.shutdown(wait=True)

if __name__ == "__main__":
    miner = OptimizedX16RMiner()
    miner.run() 