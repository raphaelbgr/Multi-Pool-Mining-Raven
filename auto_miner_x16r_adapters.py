#!/usr/bin/env python3
"""
ADAPTER-BASED X16R MINER - Ravencoin Multi-Pool Miner
Each pool has its own adapter for parsing and handling
Single CUDA hash generation → test against all pool targets
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
from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner_x16r_adapters.log'),
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
    clean_jobs: bool
    merkle_branches: List[str]
    pool_name: str

def safe_hex(val, default=None, length=None):
    """Safely convert to hex string with validation"""
    if not isinstance(val, str) or not val:
        return default
    try:
        # Remove any whitespace
        val = val.strip()
        if not val:
            return default
        # Validate hex format
        b = bytes.fromhex(val)
        if length and len(b) != length:
            return default
        return val
    except Exception:
        return default

def safe_int(val, base=16, default=None):
    """Safely convert to integer"""
    try:
        if isinstance(val, str):
            val = val.strip()
        if not val:
            return default
        return int(val, base)
    except Exception:
        return default

def ensure_list(val):
    """Ensure value is a list"""
    if isinstance(val, list):
        return val
    return []

def is_valid_hex(s, length=None):
    if not isinstance(s, str):
        return False
    try:
        b = bytes.fromhex(s)
        if length and len(b) != length:
            return False
        return True
    except Exception:
        return False

def validate_job(job):
    # Validate all required fields for a MiningJob
    if not job:
        return False
    required = [job.job_id, job.prev_hash, job.coinb1, job.coinb2, job.nbits, job.ntime, job.target]
    if not all(required):
        return False
    if not is_valid_hex(job.prev_hash, 32):
        return False
    if not is_valid_hex(job.coinb1):
        return False
    if not is_valid_hex(job.coinb2):
        return False
    if not is_valid_hex(job.nbits, 4):
        return False
    if not is_valid_hex(job.ntime, 4):
        return False
    if not is_valid_hex(job.target, 32):
        return False
    return True

class PoolAdapter(ABC):
    """Abstract base class for pool adapters"""
    
    def __init__(self, pool_config: Dict):
        self.pool_config = pool_config
        self.name = pool_config['name']
        self.host = pool_config['host']
        self.port = pool_config['port']
        self.user = pool_config['user']
        self.password = pool_config['password']
        self.socket = None
        self.buffer = ""
        
    @abstractmethod
    def connect_and_subscribe(self) -> bool:
        """Connect to pool and subscribe to mining notifications"""
        pass
        
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the pool"""
        pass
        
    @abstractmethod
    def get_job(self) -> Optional[MiningJob]:
        """Get mining job from pool"""
        pass
        
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
            self.socket = None

    def safe_json_parse(self, response: str) -> Optional[Dict]:
        """Safely parse JSON response, handling multi-line and extra data"""
        try:
            # Add to buffer
            self.buffer += response
            
            # Try to find complete JSON objects
            lines = self.buffer.split('\n')
            self.buffer = lines[-1]  # Keep incomplete line in buffer
            
            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            return None
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return None

    def validate_job_fields(self, params: List) -> bool:
        """Validate that job fields are present and valid"""
        if len(params) < 4:
            return False
        
        # Check required fields are not empty
        required_fields = [params[0], params[1], params[2], params[3]]
        for field in required_fields:
            if not field or (isinstance(field, str) and not field.strip()):
                return False
        
        return True

    def parse_mining_job(self, params: List) -> Optional[MiningJob]:
        """Parse mining job with strict validation"""
        try:
            # Validate basic structure
            if not self.validate_job_fields(params):
                logger.error(f"Invalid job fields from {self.name}: {params}")
                return None
            
            # Strict field validation with proper defaults
            job_id = safe_hex(str(params[0]), default='0')
            prev_hash = safe_hex(str(params[1]), default='00'*32, length=64)
            coinb1 = safe_hex(str(params[2]), default='')
            coinb2 = safe_hex(str(params[3]), default='')
            
            # Handle merkle_branches properly
            merkle_branches = ensure_list(params[4]) if len(params) > 4 else []
            
            # Parse other fields with validation
            version = safe_int(params[5], base=10, default=3931905) if len(params) > 5 else 3931905
            nbits = safe_hex(str(params[6]), default='1b00d5d1') if len(params) > 6 else '1b00d5d1'
            time_val = params[7] if len(params) > 7 else hex(int(time.time()))[2:]
            ntime = safe_hex(str(time_val), default=hex(int(time.time()))[2:])
            clean_jobs = params[8] if len(params) > 8 else True
            
            # Validate target
            target = safe_hex(
                self.pool_config.get('target', '00000000ffff0000000000000000000000000000000000000000000000000000'), 
                default='00000000ffff0000000000000000000000000000000000000000000000000000', 
                length=64
            )
            
            # Final validation
            if not all([job_id, prev_hash, coinb1, coinb2, nbits, ntime, target]):
                logger.error(f"Invalid job fields from {self.name}: {params}")
                return None
            
            job = MiningJob(
                job_id=job_id,
                prev_hash=prev_hash,
                coinb1=coinb1,
                coinb2=coinb2,
                merkle_branches=merkle_branches,
                version=version,
                nbits=nbits,
                ntime=ntime,
                clean_jobs=clean_jobs,
                target=target,
                pool_name=self.name
            )
            
            logger.info(f"Parsed job for {self.name}: {job}")
            return job
            
        except Exception as e:
            logger.error(f"Error parsing job from {self.name}: {e}")
            return None

# Add share submission methods to each adapter
    def submit_share(self, job_id: str, nonce: str, extranonce1: str, extranonce2: str, header_hex: str) -> bool:
        """Submit a share to the pool and return True if accepted"""
        try:
            # Create the submit message
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": [job_id, "worker", extranonce2, nonce, header_hex]
            }
            
            # Send the submit message
            self.socket.send((json.dumps(submit_msg) + "\n").encode())
            
            # Get the response
            response = self.socket.recv(4096).decode().strip()
            logging.info(f"Share submission response from {self.name}: {response}")
            
            try:
                result = json.loads(response)
                if result.get("result") == True:
                    logging.info(f"✅ Share ACCEPTED by {self.name}")
                    return True
                else:
                    logging.warning(f"❌ Share REJECTED by {self.name}: {result}")
                    return False
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response from {self.name}: {response}")
                return False
                
        except Exception as e:
            logging.error(f"Error submitting share to {self.name}: {e}")
            return False

class TwoMinersAdapter(PoolAdapter):
    """Adapter for 2Miners pool (Stratum1)"""
    
    def connect_and_subscribe(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            
            # Subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": [f"X16R-Miner/1.0", None]
            }
            self.socket.send((json.dumps(subscribe_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Subscribe response from {self.name}: {response.strip()}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    def authenticate(self) -> bool:
        try:
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [self.user, self.password]
            }
            self.socket.send((json.dumps(auth_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Auth response from {self.name}: {response.strip()}")
            
            data = self.safe_json_parse(response)
            if data:
                return data.get('result', False) == True
            return False
        except Exception as e:
            logger.error(f"Failed to authenticate to {self.name}: {e}")
            return False
    
    def get_job(self) -> Optional[MiningJob]:
        try:
            self.socket.settimeout(10)
            response = self.socket.recv(4096).decode()
            
            if not response:
                return None
                
            # Parse JSON responses
            data = self.safe_json_parse(response)
            if data and data.get('method') == 'mining.notify':
                params = data['params']
                return self.parse_mining_job(params)
                
            return None
        except Exception as e:
            logger.error(f"Error getting job from {self.name}: {e}")
            return None

    def submit_share(self, job_id: str, nonce: str, extranonce1: str, extranonce2: str, header_hex: str) -> bool:
        """Submit a share to 2Miners"""
        try:
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": [job_id, "worker", extranonce2, nonce, header_hex]
            }
            
            self.socket.send((json.dumps(submit_msg) + "\n").encode())
            response = self.socket.recv(4096).decode().strip()
            logging.info(f"2Miners share response: {response}")
            
            result = json.loads(response)
            if result.get("result") == True:
                logging.info(f"✅ Share ACCEPTED by 2Miners")
                return True
            else:
                logging.warning(f"❌ Share REJECTED by 2Miners: {result}")
                return False
                
        except Exception as e:
            logging.error(f"Error submitting share to 2Miners: {e}")
            return False

class RavenminerAdapter(PoolAdapter):
    """Adapter for Ravenminer pool (Stratum1)"""
    
    def connect_and_subscribe(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            
            # Subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": [f"X16R-Miner/1.0", None]
            }
            self.socket.send((json.dumps(subscribe_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Subscribe response from {self.name}: {response.strip()}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    def authenticate(self) -> bool:
        try:
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [self.user, self.password]
            }
            self.socket.send((json.dumps(auth_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Auth response from {self.name}: {response.strip()}")
            
            data = self.safe_json_parse(response)
            if data:
                return data.get('result', False) == True
            return False
        except Exception as e:
            logger.error(f"Failed to authenticate to {self.name}: {e}")
            return False
    
    def get_job(self) -> Optional[MiningJob]:
        try:
            self.socket.settimeout(10)
            response = self.socket.recv(4096).decode()
            
            if not response:
                return None
                
            # Parse JSON responses
            data = self.safe_json_parse(response)
            if data and data.get('method') == 'mining.notify':
                params = data['params']
                return self.parse_mining_job(params)
                
            return None
        except Exception as e:
            logger.error(f"Error getting job from {self.name}: {e}")
            return None

    def submit_share(self, job_id: str, nonce: str, extranonce1: str, extranonce2: str, header_hex: str) -> bool:
        """Submit a share to Ravenminer"""
        try:
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": [job_id, "worker", extranonce2, nonce, header_hex]
            }
            
            self.socket.send((json.dumps(submit_msg) + "\n").encode())
            response = self.socket.recv(4096).decode().strip()
            logging.info(f"Ravenminer share response: {response}")
            
            result = json.loads(response)
            if result.get("result") == True:
                logging.info(f"✅ Share ACCEPTED by Ravenminer")
                return True
            else:
                logging.warning(f"❌ Share REJECTED by Ravenminer: {result}")
                return False
                
        except Exception as e:
            logging.error(f"Error submitting share to Ravenminer: {e}")
            return False

class WoolyPoolyAdapter(PoolAdapter):
    """Adapter for WoolyPooly pool (Stratum2)"""
    
    def connect_and_subscribe(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            
            # Subscribe (Stratum2 format)
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": [f"X16R-Miner/1.0", None]
            }
            self.socket.send((json.dumps(subscribe_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Subscribe response from {self.name}: {response.strip()}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    def authenticate(self) -> bool:
        try:
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [self.user, self.password]
            }
            self.socket.send((json.dumps(auth_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Auth response from {self.name}: {response.strip()}")
            
            data = self.safe_json_parse(response)
            if data:
                return data.get('result', False) == True
            return False
        except Exception as e:
            logger.error(f"Failed to authenticate to {self.name}: {e}")
            return False
    
    def get_job(self) -> Optional[MiningJob]:
        try:
            self.socket.settimeout(10)
            response = self.socket.recv(4096).decode()
            
            if not response:
                return None
                
            # Parse JSON responses
            data = self.safe_json_parse(response)
            if data and data.get('method') == 'mining.notify':
                params = data['params']
                return self.parse_mining_job(params)
                
            return None
        except Exception as e:
            logger.error(f"Error getting job from {self.name}: {e}")
            return None

    def submit_share(self, job_id: str, nonce: str, extranonce1: str, extranonce2: str, header_hex: str) -> bool:
        """Submit a share to WoolyPooly"""
        try:
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": [job_id, "worker", extranonce2, nonce, header_hex]
            }
            
            self.socket.send((json.dumps(submit_msg) + "\n").encode())
            response = self.socket.recv(4096).decode().strip()
            logging.info(f"WoolyPooly share response: {response}")
            
            result = json.loads(response)
            if result.get("result") == True:
                logging.info(f"✅ Share ACCEPTED by WoolyPooly")
                return True
            else:
                logging.warning(f"❌ Share REJECTED by WoolyPooly: {result}")
                return False
                
        except Exception as e:
            logging.error(f"Error submitting share to WoolyPooly: {e}")
            return False

class HeroMinersAdapter(PoolAdapter):
    """Adapter for HeroMiners pool (Stratum2)"""
    
    def connect_and_subscribe(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            
            # Subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": [f"X16R-Miner/1.0", None]
            }
            self.socket.send((json.dumps(subscribe_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Subscribe response from {self.name}: {response.strip()}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    def authenticate(self) -> bool:
        try:
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [self.user, self.password]
            }
            self.socket.send((json.dumps(auth_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Auth response from {self.name}: {response.strip()}")
            
            data = self.safe_json_parse(response)
            if data:
                return data.get('result', False) == True
            return False
        except Exception as e:
            logger.error(f"Failed to authenticate to {self.name}: {e}")
            return False
    
    def get_job(self) -> Optional[MiningJob]:
        try:
            self.socket.settimeout(10)
            response = self.socket.recv(4096).decode()
            
            if not response:
                return None
                
            # Parse JSON responses
            data = self.safe_json_parse(response)
            if data and data.get('method') == 'mining.notify':
                params = data['params']
                return self.parse_mining_job(params)
                
            return None
        except Exception as e:
            logger.error(f"Error getting job from {self.name}: {e}")
            return None

    def submit_share(self, job_id: str, nonce: str, extranonce1: str, extranonce2: str, header_hex: str) -> bool:
        """Submit a share to HeroMiners"""
        try:
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": [job_id, "worker", extranonce2, nonce, header_hex]
            }
            
            self.socket.send((json.dumps(submit_msg) + "\n").encode())
            response = self.socket.recv(4096).decode().strip()
            logging.info(f"HeroMiners share response: {response}")
            
            result = json.loads(response)
            if result.get("result") == True:
                logging.info(f"✅ Share ACCEPTED by HeroMiners")
                return True
            else:
                logging.warning(f"❌ Share REJECTED by HeroMiners: {result}")
                return False
                
        except Exception as e:
            logging.error(f"Error submitting share to HeroMiners: {e}")
            return False

class NanopoolAdapter(PoolAdapter):
    """Adapter for Nanopool (Stratum1)"""
    
    def connect_and_subscribe(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            
            # Subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": [f"X16R-Miner/1.0", None]
            }
            self.socket.send((json.dumps(subscribe_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Subscribe response from {self.name}: {response.strip()}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    def authenticate(self) -> bool:
        try:
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [self.user, self.password]
            }
            self.socket.send((json.dumps(auth_msg) + "\n").encode())
            
            response = self.socket.recv(4096).decode()
            logger.info(f"Auth response from {self.name}: {response.strip()}")
            
            data = self.safe_json_parse(response)
            if data:
                return data.get('result', False) == True
            return False
        except Exception as e:
            logger.error(f"Failed to authenticate to {self.name}: {e}")
            return False
    
    def get_job(self) -> Optional[MiningJob]:
        try:
            self.socket.settimeout(10)
            response = self.socket.recv(4096).decode()
            
            if not response:
                return None
                
            # Parse JSON responses
            data = self.safe_json_parse(response)
            if data and data.get('method') == 'mining.notify':
                params = data['params']
                return self.parse_mining_job(params)
                
            return None
        except Exception as e:
            logger.error(f"Error getting job from {self.name}: {e}")
            return None

    def submit_share(self, job_id: str, nonce: str, extranonce1: str, extranonce2: str, header_hex: str) -> bool:
        """Submit a share to Nanopool"""
        try:
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": [job_id, "worker", extranonce2, nonce, header_hex]
            }
            
            self.socket.send((json.dumps(submit_msg) + "\n").encode())
            response = self.socket.recv(4096).decode().strip()
            logging.info(f"Nanopool share response: {response}")
            
            result = json.loads(response)
            if result.get("result") == True:
                logging.info(f"✅ Share ACCEPTED by Nanopool")
                return True
            else:
                logging.warning(f"❌ Share REJECTED by Nanopool: {result}")
                return False
                
        except Exception as e:
            logging.error(f"Error submitting share to Nanopool: {e}")
            return False

class X16RAdapterMiner:
    """Main miner class using adapter pattern"""
    
    def __init__(self):
        self.config = self.load_config()
        if not self.config or 'pools' not in self.config:
            raise ValueError("Invalid config: missing pools section")
        
        self.pool_adapters = []
        self.init_pool_adapters()
        self.cuda_miner_path = "x16r_miner.exe"
        
    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                logger.info("Config loaded successfully")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def init_pool_adapters(self):
        """Initialize pool adapters based on config"""
        adapter_map = {
            '2Miners': TwoMinersAdapter,
            'Ravenminer': RavenminerAdapter,
            'WoolyPooly': WoolyPoolyAdapter,
            'HeroMiners': HeroMinersAdapter,
            'Nanopool': NanopoolAdapter
        }
        
        for pool_config in self.config['pools']:
            pool_name = pool_config['name']
            if pool_name in adapter_map:
                adapter = adapter_map[pool_name](pool_config)
                self.pool_adapters.append(adapter)
                logger.info(f"Initialized {pool_name} adapter")
        
        logger.info(f"Initialized {len(self.pool_adapters)} pool adapters")
    
    def calculate_merkle_root(self, job: MiningJob, extranonce2: str) -> bytes:
        """Calculate merkle root from job data"""
        try:
            # Build coinbase transaction
            coinbase = job.coinb1 + extranonce2 + job.coinb2
            
            # Calculate merkle root
            merkle_root = hashlib.sha256(hashlib.sha256(bytes.fromhex(coinbase)).digest()).digest()
            
            # Add merkle branches if present
            for branch in job.merkle_branches:
                if branch:  # Ensure branch is not empty
                    merkle_root = hashlib.sha256(
                        hashlib.sha256(merkle_root + bytes.fromhex(branch)).digest()
                    ).digest()
            
            return merkle_root
        except Exception as e:
            logger.error(f"Failed to calculate merkle root: {e}")
            return b'\x00' * 32
    
    def build_block_header(self, job: MiningJob, nonce: int, extranonce2: str = "") -> Optional[bytes]:
        """Build block header for mining"""
        try:
            # Calculate merkle root
            merkle_root = self.calculate_merkle_root(job, extranonce2)
            
            # Build header (80 bytes)
            header = struct.pack('<I', job.version)  # Version (4 bytes)
            header += bytes.fromhex(job.prev_hash)[::-1]  # Previous hash (32 bytes, little-endian)
            header += merkle_root[::-1]  # Merkle root (32 bytes, little-endian)
            header += struct.pack('<I', int(job.ntime, 16))  # Timestamp (4 bytes)
            header += struct.pack('<I', int(job.nbits, 16))  # Bits (4 bytes)
            header += struct.pack('<I', nonce)  # Nonce (4 bytes)
            
            return header
        except Exception as e:
            logger.error(f"Failed to build header: {e}")
            return None
    
    def create_challenges_bin(self, jobs: List[Optional[MiningJob]]) -> bool:
        """Create challenges.bin file for CUDA miner"""
        try:
            challenges = []
            
            for i, job in enumerate(jobs):
                if job is None:
                    continue
                    
                # Build header for this job
                header = self.build_block_header(job, 0)
                if header:
                    challenges.append({
                        'pool_index': i,
                        'header': header,
                        'target': job.target
                    })
            
            if not challenges:
                logger.warning("No challenges available, skipping cycle")
                return False
            
            # Write challenges to file
            with open('challenges.bin', 'wb') as f:
                for challenge in challenges:
                    f.write(challenge['header'])
            
            logger.info(f"Created challenges.bin with {len(challenges)} challenges")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create challenges: {e}")
            return False
    
    def run_cuda_mining_cycle(self) -> bool:
        """Run CUDA mining cycle"""
        try:
            if not os.path.exists(self.cuda_miner_path):
                logger.error(f"CUDA miner not found: {self.cuda_miner_path}")
                return False
            
            # Run CUDA miner
            result = subprocess.run([self.cuda_miner_path], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info("CUDA mining cycle completed")
                return True
            else:
                logger.error(f"CUDA miner failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("CUDA miner timed out")
            return False
        except Exception as e:
            logger.error(f"CUDA mining error: {e}")
            return False
    
    def run_cycle(self):
        """Run one mining cycle"""
        try:
            # Get jobs from all pools
            jobs = []
            for adapter in self.pool_adapters:
                try:
                    if adapter.connect_and_subscribe():
                        if adapter.authenticate():
                            job = adapter.get_job()
                            jobs.append(job)
                        else:
                            jobs.append(None)
                    else:
                        jobs.append(None)
                except Exception as e:
                    logger.error(f"Error with {adapter.name}: {e}")
                    jobs.append(None)
            
            # Create challenges file
            if self.create_challenges_bin(jobs):
                # Run CUDA mining
                if self.run_cuda_mining_cycle():
                    logger.info("Cycle completed successfully")
                else:
                    logger.info("Cycle completed. No shares found")
            else:
                logger.info("Cycle completed. No challenges available")
                
        except Exception as e:
            logger.error(f"Cycle error: {e}")
    
    def run(self):
        """Main mining loop"""
        logger.info("Starting Adapter-Based X16R Ravencoin Miner")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("Using pool-specific adapters with shared CUDA hash testing")
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"=== Mining Cycle #{cycle_count} ===")
                
                self.run_cycle()
                
                # Wait between cycles
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping X16R miner...")
        except Exception as e:
            logger.error(f"Mining error: {e}")

def log_header_breakdown(header_bytes):
    # Assumes header is 80 bytes
    if len(header_bytes) != 80:
        logging.warning(f"Header is not 80 bytes: {len(header_bytes)}")
        return
    version = header_bytes[0:4][::-1].hex()
    prev_hash = header_bytes[4:36][::-1].hex()
    merkle_root = header_bytes[36:68][::-1].hex()
    ntime = header_bytes[68:72][::-1].hex()
    nbits = header_bytes[72:76][::-1].hex()
    nonce = header_bytes[76:80][::-1].hex()
    logging.info(f"Header breakdown:")
    logging.info(f"  version:     {version}")
    logging.info(f"  prev_hash:   {prev_hash}")
    logging.info(f"  merkle_root: {merkle_root}")
    logging.info(f"  ntime:       {ntime}")
    logging.info(f"  nbits:       {nbits}")
    logging.info(f"  nonce:       {nonce}")

# --- Update test class to log all diagnostics ---
class X16RAdapterMinerTest(X16RAdapterMiner):
    """Test version that runs for limited cycles and submits real shares"""
    
    def __init__(self, max_cycles=10):
        self.max_cycles = max_cycles
        self.cycle_count = 0
        super().__init__()
    
    def run(self):
        """Run for limited cycles with real share submission"""
        logging.info(f"Starting TEST X16R Adapter Miner (max {self.max_cycles} cycles)")
        logging.info(f"Started at: {datetime.now()}")
        
        try:
            while self.cycle_count < self.max_cycles:
                self.cycle_count += 1
                logging.info(f"=== TEST Mining Cycle #{self.cycle_count} ===")
                
                # Get jobs from all pools
                jobs = []
                for i, adapter in enumerate(self.pool_adapters):
                    try:
                        # Connect and authenticate first
                        if adapter.connect_and_subscribe() and adapter.authenticate():
                            job = adapter.get_job()
                            if job and validate_job(job):
                                jobs.append((i, job))
                                logging.info(f"Valid job from {adapter.name}: {job.job_id}")
                            else:
                                logging.warning(f"Invalid job from {adapter.name}, skipping")
                        else:
                            logging.warning(f"Failed to connect/authenticate to {adapter.name}")
                    except Exception as e:
                        logging.error(f"Error getting job from {adapter.name}: {e}")
                
                if not jobs:
                    logging.warning("No valid jobs available, skipping cycle")
                    time.sleep(1)
                    continue
                
                # Create challenges for valid jobs
                challenges = []
                for pool_idx, job in jobs:
                    try:
                        header = self.build_block_header(job, 0)
                        if header:
                            challenges.append((pool_idx, job, header))
                            logging.info(f"Created challenge for {self.pool_adapters[pool_idx].name}")
                    except Exception as e:
                        logging.error(f"Failed to build header for {self.pool_adapters[pool_idx].name}: {e}")
                
                if not challenges:
                    logging.warning("No challenges available, skipping cycle")
                    time.sleep(1)
                    continue
                
                # Write challenges to file
                with open("challenges.bin", "wb") as f:
                    f.write(struct.pack("<I", len(challenges)))
                    for pool_idx, job, header in challenges:
                        f.write(struct.pack("<I", pool_idx))
                        f.write(header)
                
                logging.info(f"Created challenges.bin with {len(challenges)} challenges")
                
                # Simulate finding a share and submit it
                for pool_idx, job, header in challenges:
                    # Create a test share with nonce 0x12345678
                    test_nonce = "12345678"
                    header_bytes = bytearray(header)
                    header_bytes[76:80] = struct.pack("<I", int(test_nonce, 16))  # Set nonce
                    header_hex = header_bytes.hex()
                    
                    # Submit the share
                    adapter = self.pool_adapters[pool_idx]
                    success = adapter.submit_share(
                        job_id=job.job_id,
                        nonce=test_nonce,
                        extranonce1="worker",
                        extranonce2="00000000",
                        header_hex=header_hex
                    )
                    
                    if success:
                        logging.info(f"✅ Share submitted successfully to {adapter.name}")
                    else:
                        logging.warning(f"❌ Share submission failed to {adapter.name}")
                
                logging.info("CUDA mining cycle completed (simulated)")
                time.sleep(1)
                logging.info("Cycle completed successfully")
                time.sleep(1)
            
            logging.info(f"Test completed after {self.max_cycles} cycles")
            
        except KeyboardInterrupt:
            logging.info("Test stopped by user")
        except Exception as e:
            logging.error(f"Error in test: {e}")
            import traceback
            traceback.print_exc()

# --- Update main execution to use --test flag ---
if __name__ == "__main__":
    print("Starting X16R Adapter Miner...")
    try:
        cuda_miner_path = "x16r_miner.exe"
        test_mode = '--test' in sys.argv
        if not os.path.exists(cuda_miner_path):
            print(f"Warning: CUDA miner '{cuda_miner_path}' not found")
            print("Running in test mode with simulated mining")
            test_mode = True
        if test_mode:
            miner = X16RAdapterMinerTest(max_cycles=10)
            print("Test miner initialized successfully")
            miner.run()
        else:
            miner = X16RAdapterMiner()
            print("Miner initialized successfully")
            miner.run()
    except Exception as e:
        print(f"Error starting miner: {e}")
        import traceback
        traceback.print_exc() 