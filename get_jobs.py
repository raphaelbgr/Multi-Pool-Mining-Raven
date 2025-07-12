import socket
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            return config
    except Exception as e:
        print(f"Config error: {str(e)}")
        raise

def create_socket_connection(host, port):
    """Create and return a socket connection with timeout"""
    sock = socket.create_connection((host, port), timeout=TIMEOUT)
    sock.settimeout(TIMEOUT)
    return sock

def send_stratum_message(sock, method, params, msg_id=1):
    """Send Stratum protocol message and return response"""
    msg = json.dumps({
        "id": msg_id,
        "method": method,
        "params": params
    }).encode() + b"\n"
    sock.sendall(msg)
    return sock.recv(4096).decode()

def process_stratum_job(pool, job_data):
    """Process Stratum job notification into mining job format"""
    try:
        params = job_data['params']
        return {
            'pool_index': pool['index'],
            'pool_name': pool['name'],
            'job_id': params[0],
            'header_hash': params[1],
            'target': params[3],
            'ntime': params[6] if len(params) > 6 else None
        }
    except (KeyError, IndexError) as e:
        print(f"Invalid job format from {pool['name']}: {str(e)}")
        return None

def handle_pool_connection(pool):
    """Handle complete pool connection and job retrieval"""
    sock = None
    try:
        sock = create_socket_connection(pool['host'], pool['port'])
        
        # Special handling for Nanopool
        if "nanopool" in pool['name'].lower():
            # Nanopool requires immediate subscription without params
            subscribe_resp = send_stratum_message(sock, "mining.subscribe", [])
            print(f"{pool['name']} subscribe: {subscribe_resp.strip()}")
            
            # Nanopool sends target immediately after subscription
            initial_data = sock.recv(4096).decode()
            print(f"{pool['name']} initial data: {initial_data.strip()}")
            
            # Then proceed with auth
            auth_resp = send_stratum_message(sock, "mining.authorize", 
                                           [pool['user'], pool['password']], 2)
            print(f"{pool['name']} auth: {auth_resp.strip()}")
        else:
            # Standard Stratum flow for other pools
            subscribe_resp = send_stratum_message(sock, "mining.subscribe", [])
            print(f"{pool['name']} subscribe: {subscribe_resp.strip()}")
            
            auth_resp = send_stratum_message(sock, "mining.authorize", 
                                           [pool['user'], pool['password']], 2)
            print(f"{pool['name']} auth: {auth_resp.strip()}")
        
        # Listen for jobs with extended timeout for Nanopool
        print(f"⏳ Waiting for job from {pool['name']}...")
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
                        return process_stratum_job(pool, msg)
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"❌ {pool['name']} error: {str(e)}")
        return None
    finally:
        if sock:
            try:
                sock.close()
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
            header_bytes = bytes.fromhex(job['header_hash'])
            headers.extend(struct.pack("<I", len(header_bytes)))
            headers.extend(header_bytes)
            targets.extend(bytes.fromhex(job['target']))
        else:
            # Empty slot
            headers.extend(bytes(4 + 32))
            targets.extend(bytes(32))
    
    return headers + targets

def save_results(jobs, pool_count):
    """Save results to files"""
    binary_data = generate_binary_data(jobs, pool_count)
    
    with open(HEADERS_FILE, "wb") as f:
        f.write(binary_data)
    
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)
    
    print(f"\n✅ Saved {len(binary_data)} bytes to {HEADERS_FILE}")
    print(f"✅ Saved {len(jobs)} jobs to {JOBS_FILE}")

def get_all_jobs():
    """Main function to retrieve jobs from all pools"""
    config = load_config()
    pool_count = len(config['pools'])
    
    # Add index to each pool config
    for i, pool in enumerate(config['pools']):
        pool['index'] = i
    
    jobs = []
    print(f"\n🔍 Connecting to {pool_count} pools...")
    
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
                    print(f"✅ {pool['name']}: Got job {job['job_id']}")
                else:
                    print(f"⚠️ {pool['name']}: No job received")
            except Exception as e:
                print(f"❌ {pool['name']} thread error: {str(e)}")
    
    save_results(jobs, pool_count)
    return jobs

if __name__ == "__main__":
    print("⛏️ Starting mining job collection...")
    start_time = time.time()
    
    try:
        jobs = get_all_jobs()
        success_rate = len(jobs) / len(load_config()['pools']) * 100
        print(f"\n🎉 Successfully collected {len(jobs)} jobs ({success_rate:.1f}%)")
        print(f"⏱️  Completed in {time.time()-start_time:.2f} seconds")
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")
        exit(1)