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
        """Submit a share to the pool using T-Rex format and return True if accepted"""
        try:
            # Extract solution (merkle_root) from header
            # Header format: version(4) + prev_hash(32) + merkle_root(32) + ntime(4) + nbits(4) + nonce(4)
            header_bytes = bytes.fromhex(header_hex)
            if len(header_bytes) != 80:
                logging.error(f"Invalid header length: {len(header_bytes)}")
                return False
            
            # Extract merkle_root (bytes 36-68) - this is the solution
            merkle_root = header_bytes[36:68][::-1].hex()  # Reverse for little-endian
            
            # Create the submit message in T-Rex format
            submit_msg = {
                "id": 4,
                "method": "mining.submit",
                "params": [
                    self.user,                    # username (1st parameter)
                    job_id,                       # job_id (2nd parameter)
                    f"0x{extranonce2}",          # extranonce2 (3rd parameter) - with 0x prefix
                    f"0x{self.current_job.prev_hash}",  # prev_hash (4th parameter) - with 0x prefix
                    f"0x{merkle_root}"           # solution/merkle_root (5th parameter) - with 0x prefix
                ],
                "worker": "worker1"
            }
            
            # Send the submit message
            self.socket.send((json.dumps(submit_msg) + "\n").encode())
            
            # Get the response
            response = self.socket.recv(4096).decode().strip()
            logging.info(f"Share submission response from {self.name}: {response}")
            
            try:
                data = json.loads(response)
                if data.get('result') == True:
                    logging.info(f"✅ Share ACCEPTED by {self.name}")
                    return True
                else:
                    error = data.get('error', ['Unknown error'])
                    logging.warning(f"❌ Share REJECTED by {self.name}: {error}")
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
        """Create separate challenges.bin files for each pool"""
        try:
            # Filter out None jobs
            valid_jobs = [job for job in jobs if job is not None]
            if not valid_jobs:
                logging.warning("No valid jobs to create challenges")
                return False
            
            # Group jobs by pool
            pool_jobs = {}
            for job in valid_jobs:
                if job.pool_name not in pool_jobs:
                    pool_jobs[job.pool_name] = []
                pool_jobs[job.pool_name].append(job)
            
            # Create separate header files for each pool
            total_headers = 0
            for pool_name, pool_job_list in pool_jobs.items():
                try:
                    headers = []
                    for job in pool_job_list:
                        try:
                            # Build header for this job
                            header = self.build_block_header(job, 0)  # Use nonce 0 for template
                            if header:
                                headers.append(header)
                                logging.debug(f"Added header for {pool_name} job {job.job_id}")
                        except Exception as e:
                            logging.error(f"Error building header for {pool_name}: {e}")
                    
                    if headers:
                        # Write headers to pool-specific file
                        filename = f"challenges_{pool_name.lower().replace(' ', '_')}.bin"
                        with open(filename, "wb") as f:
                            for header in headers:
                                f.write(header)
                        
                        logging.info(f"Created {filename} with {len(headers)} challenges for {pool_name}")
                        total_headers += len(headers)
                
                except Exception as e:
                    logging.error(f"Error creating challenges for {pool_name}: {e}")
            
            if total_headers > 0:
                logging.info(f"Created {len(pool_jobs)} pool-specific challenge files with {total_headers} total challenges")
                return True
            else:
                logging.error("No valid headers created for any pool")
                return False
            
        except Exception as e:
            logging.error(f"Error creating challenges.bin: {e}")
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

# Add comprehensive diagnostic logging functions
def log_share_diagnostics(job, header_bytes, nonce, extranonce1, extranonce2, pool_name):
    """Log complete share submission diagnostics"""
    logging.info(f"=== SHARE DIAGNOSTICS FOR {pool_name} ===")
    
    # Log all job fields
    logging.info(f"Job ID: {job.job_id}")
    logging.info(f"Prev Hash: {job.prev_hash}")
    logging.info(f"Coinb1: {job.coinb1}")
    logging.info(f"Coinb2: {job.coinb2}")
    logging.info(f"Version: {job.version} (0x{job.version:08x})")
    logging.info(f"Nbits: {job.nbits}")
    logging.info(f"Ntime: {job.ntime}")
    logging.info(f"Target: {job.target}")
    logging.info(f"Merkle Branches: {job.merkle_branches}")
    
    # Log header breakdown
    if len(header_bytes) == 80:
        version = header_bytes[0:4][::-1].hex()
        prev_hash = header_bytes[4:36][::-1].hex()
        merkle_root = header_bytes[36:68][::-1].hex()
        ntime = header_bytes[68:72][::-1].hex()
        nbits = header_bytes[72:76][::-1].hex()
        nonce_bytes = header_bytes[76:80][::-1].hex()
        
        logging.info(f"Header Breakdown:")
        logging.info(f"  Version: {version}")
        logging.info(f"  Prev Hash: {prev_hash}")
        logging.info(f"  Merkle Root: {merkle_root}")
        logging.info(f"  Ntime: {ntime}")
        logging.info(f"  Nbits: {nbits}")
        logging.info(f"  Nonce: {nonce_bytes}")
    
    # Log full header
    logging.info(f"Full Header (80 bytes): {header_bytes.hex()}")
    
    # Log submission parameters
    logging.info(f"Submission Parameters:")
    logging.info(f"  Nonce: {nonce}")
    logging.info(f"  Extranonce1: {extranonce1}")
    logging.info(f"  Extranonce2: {extranonce2}")
    
    # Log what JSON-RPC message would be sent
    submit_msg = {
        "id": 4,
        "method": "mining.submit",
        "params": [job.job_id, extranonce2, job.ntime, nonce]
    }
    logging.info(f"JSON-RPC Message: {json.dumps(submit_msg)}")
    
    logging.info(f"=== END DIAGNOSTICS ===")

def log_pool_comparison(pool_name, job, reference_data=None):
    """Log comparison with reference miner data"""
    logging.info(f"=== POOL COMPARISON: {pool_name} ===")
    logging.info(f"Our Job Data:")
    logging.info(f"  Job ID: {job.job_id}")
    logging.info(f"  Prev Hash: {job.prev_hash}")
    logging.info(f"  Version: {job.version}")
    logging.info(f"  Nbits: {job.nbits}")
    logging.info(f"  Ntime: {job.ntime}")
    
    if reference_data:
        logging.info(f"Reference Miner Data:")
        logging.info(f"  Job ID: {reference_data.get('job_id', 'N/A')}")
        logging.info(f"  Prev Hash: {reference_data.get('prev_hash', 'N/A')}")
        logging.info(f"  Version: {reference_data.get('version', 'N/A')}")
        logging.info(f"  Nbits: {reference_data.get('nbits', 'N/A')}")
        logging.info(f"  Ntime: {reference_data.get('ntime', 'N/A')}")
    
    logging.info(f"=== END COMPARISON ===")

# --- Update test class to log all diagnostics ---
class X16RAdapterMinerTest(X16RAdapterMiner):
    """Test version that runs for limited cycles and submits real shares with full diagnostics"""
    
    def __init__(self, max_cycles=5):
        self.max_cycles = max_cycles
        self.cycle_count = 0
        super().__init__()
    
    def run(self):
        """Run for limited cycles with real share submission and diagnostics"""
        logging.info(f"Starting TEST X16R Adapter Miner (max {self.max_cycles} cycles)")
        logging.info(f"Started at: {datetime.now()}")
        logging.info("DIAGNOSTIC MODE: Will log all share details and submit to pools")
        
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
                            if job:
                                jobs.append((i, adapter, job))
                                logging.info(f"Got job from {adapter.name}: {job.job_id}")
                            else:
                                logging.warning(f"No job from {adapter.name}")
                        else:
                            logging.warning(f"Failed to connect/authenticate to {adapter.name}")
                    except Exception as e:
                        logging.error(f"Error getting job from {adapter.name}: {e}")
                
                if not jobs:
                    logging.warning("No jobs received, skipping cycle")
                    time.sleep(1)
                    continue
                
                # For each job, create a test share and submit it
                for pool_idx, adapter, job in jobs:
                    try:
                        logging.info(f"Processing job {job.job_id} from {adapter.name}")
                        
                        # Build header (simulate what your miner does)
                        header_bytes = self.build_test_header(job)
                        
                        # Create test nonce and extranonce
                        test_nonce = "12345678"
                        extranonce1 = adapter.extranonce1 if hasattr(adapter, 'extranonce1') else "00000000"
                        extranonce2 = "00000000"
                        
                        # Log complete diagnostics
                        log_share_diagnostics(job, header_bytes, test_nonce, extranonce1, extranonce2, adapter.name)
                        
                        # Submit the share to the pool
                        logging.info(f"Submitting test share to {adapter.name}...")
                        success = adapter.submit_share(job.job_id, test_nonce, extranonce1, extranonce2, header_bytes.hex())
                        
                        if success:
                            logging.info(f"✅ Share accepted by {adapter.name}")
                        else:
                            logging.warning(f"❌ Share rejected by {adapter.name}")
                            
                    except Exception as e:
                        logging.error(f"Error processing job from {adapter.name}: {e}")
                
                logging.info(f"Cycle {self.cycle_count} completed")
                time.sleep(2)  # Wait between cycles
                
        except KeyboardInterrupt:
            logging.info("Test miner stopped by user")
        except Exception as e:
            logging.error(f"Test miner error: {e}")
            import traceback
            traceback.print_exc()
    
    def build_test_header(self, job):
        """Build a test header for diagnostics"""
        try:
            # Convert version to little-endian bytes
            version_bytes = struct.pack('<I', job.version)
            
            # Convert prev_hash to little-endian bytes (reverse the hex string)
            prev_hash_bytes = bytes.fromhex(job.prev_hash)[::-1]
            
            # Calculate merkle root (simplified - you'll need proper calculation)
            coinbase = job.coinb1 + "00000000" + job.coinb2  # Add extranonce2
            merkle_root = hashlib.sha256(hashlib.sha256(bytes.fromhex(coinbase)).digest()).digest()
            
            # Convert ntime to little-endian bytes
            ntime_int = int(job.ntime, 16)
            ntime_bytes = struct.pack('<I', ntime_int)
            
            # Convert nbits to little-endian bytes
            nbits_int = int(job.nbits, 16)
            nbits_bytes = struct.pack('<I', nbits_int)
            
            # Test nonce
            nonce_bytes = struct.pack('<I', 0x12345678)
            
            # Build 80-byte header
            header = version_bytes + prev_hash_bytes + merkle_root + ntime_bytes + nbits_bytes + nonce_bytes
            
            return header
            
        except Exception as e:
            logging.error(f"Error building test header: {e}")
            return b'\x00' * 80  # Return empty header on error

def submit_known_valid_share():
    """Test share submission format with current job data"""
    import socket
    import json
    import time
    logging.info("Testing share submission format with current job data...")
    
    # 2Miners connection info
    host = "rvn.2miners.com"
    port = 6060
    user = "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"
    password = "x"
    worker = "raphael-desktop"
    
    try:
        # Connect to pool
        s = socket.create_connection((host, port), timeout=10)
        s.settimeout(30)
        f = s.makefile("rw")
        
        # Subscribe
        subscribe = {"id": 1, "method": "mining.subscribe", "params": ["t-rex/0.26.8"]}
        s.send((json.dumps(subscribe) + "\n").encode())
        resp = f.readline()
        logging.info(f"Subscribe response: {resp.strip()}")
        
        # Authorize
        authorize = {"id": 2, "method": "mining.authorize", "params": [user, password], "worker": worker}
        s.send((json.dumps(authorize) + "\n").encode())
        resp = f.readline()
        logging.info(f"Authorize response: {resp.strip()}")
        
        # Wait for current job
        job_data = None
        for _ in range(5):
            try:
                line = f.readline()
                if not line:
                    break
                logging.info(f"Pool message: {line.strip()}")
                if "mining.notify" in line:
                    # Parse the job data
                    try:
                        data = json.loads(line)
                        if data.get("method") == "mining.notify":
                            params = data.get("params", [])
                            if len(params) >= 7:
                                job_data = {
                                    "job_id": params[0],
                                    "prev_hash": params[1],
                                    "coinb1": params[2],
                                    "coinb2": params[3],
                                    "version": params[5],
                                    "nbits": params[6]
                                }
                                logging.info(f"Got current job: {job_data}")
                                break
                    except:
                        pass
            except Exception:
                break
        
        if job_data:
            # Submit a test share with current job data but fixed nonce
            submit_msg = {
                "id": 88181504,
                "method": "mining.submit",
                "params": [
                    job_data["job_id"],
                    "00000000",  # extranonce2
                    "6875a2e0",  # ntime (current time)
                    "12345678"   # test nonce
                ],
                "worker": worker
            }
            s.send((json.dumps(submit_msg) + "\n").encode())
            logging.info(f"Submitted test share: {json.dumps(submit_msg)}")
            
            # Get the response
            try:
                resp = f.readline()
                logging.info(f"Share submission response: {resp.strip()}")
            except Exception as e:
                logging.error(f"Timeout waiting for response: {e}")
        else:
            logging.error("No job data received")
        
        s.close()
        
    except Exception as e:
        logging.error(f"Error testing share submission: {e}")
        import traceback
        traceback.print_exc()

# --- Update main execution to use --test flag ---
if __name__ == "__main__":
    print("Starting X16R Adapter Miner...")
    try:
        cuda_miner_path = "x16r_miner.exe"
        test_mode = '--test' in sys.argv
        submit_known_share = '--submit-known-share' in sys.argv
        
        if submit_known_share:
            print("Testing share submission format...")
            submit_known_valid_share()
            print("Share submission test completed.")
            exit(0)  # Exit after testing
        elif not os.path.exists(cuda_miner_path):
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