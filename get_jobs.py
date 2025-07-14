import socket
import json
import struct
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('get_jobs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
HEADER_SIZE = 80
TARGET_SIZE = 32
CONFIG_FILE = "config.json"
HEADERS_FILE = "headers.bin"
JOBS_FILE = "jobs.json"
TIMEOUT = 10

def load_config():
    """Load pool configuration from JSON file"""
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
            # Validate config structure
            required_keys = {'host', 'port', 'user', 'password', 'name'}
            for pool in config['pools']:
                if not all(k in pool for k in required_keys):
                    raise ValueError("Invalid pool configuration")
            logger.info(f"[OK] Config loaded: {len(config['pools'])} pools configured")
            return config
    except Exception as e:
        logger.error(f"Config error: {str(e)}")
        raise

def create_socket_connection(host, port):
    """Create and return a socket connection with timeout"""
    logger.debug(f"Connecting to {host}:{port}...")
    sock = socket.create_connection((host, port), timeout=TIMEOUT)
    sock.settimeout(TIMEOUT)
    logger.debug(f"[OK] Connected to {host}:{port}")
    return sock

def send_stratum_message(sock, method, params, msg_id=1):
    """Send Stratum protocol message and return response"""
    msg = json.dumps({
        "id": msg_id,
        "method": method,
        "params": params
    }).encode() + b"\n"
    logger.debug(f"Sending: {method} with params: {params}")
    sock.sendall(msg)
    response = sock.recv(4096).decode()
    logger.debug(f"Received: {response.strip()}")
    return response

def process_stratum_job(pool, job_data):
    """Process Stratum job notification into mining job format"""
    try:
        params = job_data['params']
        job = {
            'pool_index': pool['index'],
            'pool_name': pool['name'],
            'job_id': params[0],
            'header_hash': params[1],
            'target': params[3],
            'ntime': params[6] if len(params) > 6 else None
        }
        logger.info(f"[OK] Processed job from {pool['name']}: {job['job_id']}")
        logger.debug(f"  Header: {job['header_hash'][:16]}...")
        logger.debug(f"  Target: {job['target'][:16]}...")
        return job
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid job format from {pool['name']}: {str(e)}")
        return None

def handle_pool_connection(pool):
    """Handle complete pool connection and job retrieval"""
    sock = None
    try:
        logger.info(f"[CONNECT] Connecting to {pool['name']} ({pool['host']}:{pool['port']})")
        sock = create_socket_connection(pool['host'], pool['port'])
        
        # Special handling for Nanopool
        if "nanopool" in pool['name'].lower():
            logger.debug(f"Using Nanopool-specific protocol for {pool['name']}")
            # Nanopool requires immediate subscription without params
            subscribe_resp = send_stratum_message(sock, "mining.subscribe", [])
            logger.info(f"{pool['name']} subscribe: {subscribe_resp.strip()}")
            
            # Nanopool sends target immediately after subscription
            initial_data = sock.recv(4096).decode()
            logger.info(f"{pool['name']} initial data: {initial_data.strip()}")
            
            # Then proceed with auth
            auth_resp = send_stratum_message(sock, "mining.authorize", 
                                           [pool['user'], pool['password']], 2)
            logger.info(f"{pool['name']} auth: {auth_resp.strip()}")
        else:
            # Standard Stratum flow for other pools
            logger.debug(f"Using standard Stratum protocol for {pool['name']}")
            subscribe_resp = send_stratum_message(sock, "mining.subscribe", [])
            logger.info(f"{pool['name']} subscribe: {subscribe_resp.strip()}")
            
            auth_resp = send_stratum_message(sock, "mining.authorize", 
                                           [pool['user'], pool['password']], 2)
            logger.info(f"{pool['name']} auth: {auth_resp.strip()}")
        
        # Listen for jobs with extended timeout for Nanopool
        logger.info(f"[WAIT] Waiting for job from {pool['name']}...")
        timeout = 20 if "nanopool" in pool['name'].lower() else TIMEOUT
        sock.settimeout(timeout)
        
        while True:
            data = sock.recv(4096).decode()
            if not data:
                continue
                
            for line in data.splitlines():
                try:
                    msg = json.loads(line)
                    if msg.get('method') == 'mining.notify':
                        logger.info(f"[JOB] Received job notification from {pool['name']}")
                        return process_stratum_job(pool, msg)
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        logger.error(f"❌ {pool['name']} error: {str(e)}")
        return None
    finally:
        if sock:
            try:
                sock.close()
                logger.debug(f"[CLOSE] Closed connection to {pool['name']}")
            except:
                pass

def generate_binary_data(jobs, pool_count):
    """Generate binary headers and targets data"""
    headers = bytearray()
    targets = bytearray()
    
    # Create slots for all pools, even failed ones
    job_dict = {job['pool_index']: job for job in jobs}
    for i in range(pool_count):
        if i in job_dict:
            job = job_dict[i]
            
            # Construct full 80-byte Ravencoin header
            full_header = construct_ravencoin_header(job)
            
            headers.extend(struct.pack("<I", len(full_header)))  # Should be 80
            headers.extend(full_header)
            targets.extend(bytes.fromhex(job['target']))
            logger.debug(f"Pool {i}: Added {len(full_header)}-byte header and target data")
        else:
            # Empty slot - 4 bytes header length + 80 bytes header + 32 bytes target
            headers.extend(struct.pack("<I", 80))  # Header length
            headers.extend(bytes(80))  # Empty 80-byte header
            targets.extend(bytes(32))  # Empty target
            logger.debug(f"Pool {i}: Empty slot (no job)")
    
    return headers + targets

def construct_ravencoin_header(job):
    """Construct proper 80-byte Ravencoin header from Stratum job data"""
    
    # Ravencoin header structure (80 bytes):
    # - Version (4 bytes): 0x20000000
    # - Previous block hash (32 bytes): zeros for now
    # - Merkle root (32 bytes): from header_hash
    # - Timestamp (4 bytes): from job ntime
    # - Bits (4 bytes): zeros for now
    # - Nonce (4 bytes): will be set by miner
    
    header_bytes = bytes.fromhex(job['header_hash'])
    
    if len(header_bytes) == 32:
        # This is likely the merkle root, construct proper header
        header = bytearray(80)
        
        # Version (4 bytes) - Ravencoin version
        header[0:4] = struct.pack("<I", 0x20000000)
        
        # Previous block hash (32 bytes) - use zeros for now
        header[4:36] = bytes(32)
        
        # Merkle root (32 bytes) - use the header_hash
        header[36:68] = header_bytes
        
        # Timestamp (4 bytes) - use job ntime if available
        if job.get('ntime'):
            try:
                ntime_int = int(job['ntime'], 16)
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
        # Unknown format, return as-is
        return header_bytes

def save_results(jobs, pool_count):
    """Save results to files"""
    binary_data = generate_binary_data(jobs, pool_count)
    
    with open(HEADERS_FILE, "wb") as f:
        f.write(binary_data)
    
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)
    
    logger.info(f"[OK] Saved {len(binary_data)} bytes to {HEADERS_FILE}")
    logger.info(f"[OK] Saved {len(jobs)} jobs to {JOBS_FILE}")

def get_all_jobs():
    """Main function to retrieve jobs from all pools"""
    config = load_config()
    pool_count = len(config['pools'])
    
    # Add index to each pool config
    for i, pool in enumerate(config['pools']):
        pool['index'] = i
    
    jobs = []
    logger.info(f"\n🔍 Connecting to {pool_count} pools...")
    
    with ThreadPoolExecutor(max_workers=pool_count) as executor:
        future_to_pool = {
            executor.submit(handle_pool_connection, pool): pool 
            for pool in config['pools']
        }
        
        for future in as_completed(future_to_pool):
            pool = future_to_pool[future]
            try:
                job = future.result()
                if job:
                    jobs.append(job)
                    logger.info(f"[OK] {pool['name']}: Got job {job['job_id']}")
                else:
                    logger.warning(f"[WARN] {pool['name']}: No job received")
            except Exception as e:
                logger.error(f"[ERROR] {pool['name']} thread error: {str(e)}")
    
    save_results(jobs, pool_count)
    return jobs

if __name__ == "__main__":
    logger.info("[START] Starting mining job collection...")
    start_time = time.time()
    
    try:
        jobs = get_all_jobs()
        success_rate = len(jobs) / len(load_config()['pools']) * 100
        logger.info(f"\n[SUCCESS] Successfully collected {len(jobs)} jobs ({success_rate:.1f}%)")
        logger.info(f"[TIME] Completed in {time.time()-start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"\n[ERROR] Critical error: {str(e)}")
        exit(1)