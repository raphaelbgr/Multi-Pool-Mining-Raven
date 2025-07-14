#!/usr/bin/env python3
"""
Multi-Pool Proxy for Ravencoin Mining
Uses existing KawPOW miner and distributes shares to multiple pools
"""

import json
import socket
import threading
import time
import logging
import signal
import sys
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_pool_proxy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PoolProxy:
    def __init__(self, config_file='config.json'):
        self.config = self.load_config(config_file)
        self.pools = self.config['pools']
        self.proxy_port = 4444  # Port for miner to connect to
        self.pool_connections = {}
        self.current_job = None
        self.job_counter = 0
        self.miner_socket = None
        self.primary_adapter = None
        self.running = True
        self.pool_debug_info = {}  # Store debug info for each pool
        self.job_storage = {}  # Store original job data by job_id
        
        # Enhanced learning system for nonce extraction
        self.learning_complete = {}
        
        # Job freshness tracking
        self.job_age_limit = 10  # Further reduce to 10 seconds for better timing
        self.max_stored_jobs = 5  # Reduce from 10 to 5 to keep only fresh jobs
        
        # Job ID mapping system
        self.job_id_counter = 0
        self.trex_to_pool_job_mapping = {}  # Maps T-Rex job IDs to original pool job IDs
        self.pool_to_trex_job_mapping = {}  # Maps original pool job IDs to T-Rex job IDs
        
        # Timing improvements
        self.last_job_time = 0
        self.job_update_frequency = 2  # Minimum seconds between job updates
        self.share_submission_lock = threading.Lock()  # Prevent concurrent submissions
        
        logger.info(f"PoolProxy initialized with {len(self.pools)} pools")
        logger.info(f"Job age limit: {self.job_age_limit}s, Max stored jobs: {self.max_stored_jobs}")
        logger.info(f"Job update frequency: {self.job_update_frequency}s")

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals"""
        logger.info("Received shutdown signal, cleaning up...")
        self.running = False
        
        # Close all pool connections
        for pool_name, adapter in self.pool_connections.items():
            try:
                if adapter and adapter.connected:
                    adapter.close()
                    logger.info(f"Closed connection to {pool_name}")
            except:
                pass
        
        # Close miner socket
        if self.miner_socket:
            try:
                self.miner_socket.close()
                logger.info("Closed miner connection")
            except:
                pass
        
        logger.info("Proxy shutdown complete")
        sys.exit(0)
        
    def start_proxy_server(self):
        """Start proxy server for miner to connect to"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('localhost', self.proxy_port))
        server.listen(1)
        
        logger.info(f"Multi-pool proxy listening on port {self.proxy_port}")
        logger.info("Configure your miner to connect to: stratum+tcp://localhost:4444")
        logger.info("Use any wallet address - the proxy will handle multi-pool distribution")
        logger.info("Press Ctrl+C to stop the proxy")
        
        while self.running:
            try:
                server.settimeout(1)  # Allow checking self.running periodically
                client, addr = server.accept()
                logger.info(f"Miner connected from {addr}")
                
                # Handle miner connection
                miner_thread = threading.Thread(target=self.handle_miner, args=(client,))
                miner_thread.daemon = True
                miner_thread.start()
                
            except socket.timeout:
                continue  # Check self.running again
            except Exception as e:
                if self.running:  # Only log errors if we're still running
                    logger.error(f"Error accepting connection: {e}")
        
        server.close()
    
    def handle_miner(self, client_socket):
        """Handle communication with the miner"""
        self.miner_socket = client_socket  # Store for sending jobs
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Parse miner request
                message = json.loads(data.decode().strip())
                response = self.process_miner_request(message)
                
                if response:
                    client_socket.send((json.dumps(response) + "\n").encode())
                    
        except Exception as e:
            logger.error(f"Error handling miner: {e}")
        finally:
            client_socket.close()
            self.miner_socket = None
    
    def process_miner_request(self, message):
        """Process request from miner"""
        method = message.get('method')
        
        if method == 'mining.subscribe':
            return {
                "id": message.get('id'),
                "result": [["mining.notify", "subscription_id"], "ae", 4],
                "error": None
            }
        
        elif method == 'mining.authorize':
            logger.info("Miner authorized")
            
            # Start job fetching from pools
            self.start_pool_jobs()
            
            # Send initial job to miner
            self.send_job_to_miner()
            
            return {
                "id": message.get('id'),
                "result": True,
                "error": None
            }
        
        elif method == 'mining.submit':
            # This is where we distribute the share to multiple pools
            return self.handle_share_submission(message)
        
        return None
    
    def start_pool_jobs(self):
        """Start fetching jobs from all pools"""
        for pool in self.pools:
            thread = threading.Thread(target=self.pool_job_fetcher, args=(pool,))
            thread.daemon = True
            thread.start()
    
    def pool_job_fetcher(self, pool_config):
        """Continuously fetch jobs from a pool"""
        logger.info(f"Started job fetcher for {pool_config['name']}")
        
        try:
            from adapters.factory import AdapterFactory
            adapter = AdapterFactory.create_adapter(pool_config)
            adapter.connect()
            
            if adapter.connected:
                logger.info(f"Connected to {pool_config['name']} for job fetching")
                
                # Use first successfully connected pool as primary job source
                if not hasattr(self, 'primary_adapter') or not self.primary_adapter:
                    self.primary_adapter = adapter
                    logger.info(f"Set {pool_config['name']} as primary job source")
                    
                    # Start job listening thread for primary pool
                    job_thread = threading.Thread(target=self.listen_for_jobs, args=(adapter, pool_config))
                    job_thread.daemon = True
                    job_thread.start()
                    
                # Store adapter for share submission
                self.pool_connections[pool_config['name']] = adapter
                
            else:
                logger.error(f"Failed to connect to {pool_config['name']}")
                
        except Exception as e:
            logger.error(f"Error connecting to {pool_config['name']}: {e}")
    
    def listen_for_jobs(self, adapter, pool_config):
        """Listen for job notifications from primary pool"""
        logger.info(f"Starting job listener for {pool_config['name']}")
        
        while self.running:
            try:
                # Set a timeout so we can check self.running periodically
                adapter.sock.settimeout(1)  # Reduced timeout for better responsiveness
                
                # Listen for incoming messages
                data = adapter.sock.recv(4096)
                if not data:
                    continue
                    
                for line in data.decode().split('\n'):
                    if not line.strip():
                        continue
                        
                    try:
                        message = json.loads(line)
                        
                        # Forward job notifications to miner
                        if message.get('method') == 'mining.notify':
                            self.forward_job_to_miner(message)
                            
                        # Forward difficulty changes
                        elif message.get('method') == 'mining.set_difficulty':
                            self.forward_to_miner(message)
                            
                        # Forward target changes  
                        elif message.get('method') == 'mining.set_target':
                            self.forward_to_miner(message)
                            
                    except json.JSONDecodeError:
                        continue
                        
            except socket.timeout:
                continue  # Check self.running again
            except Exception as e:
                if self.running:  # Only log errors if we're still running
                    logger.error(f"Error listening for jobs from {pool_config['name']}: {e}")
                time.sleep(1)
                break
        
        logger.info(f"Job listener for {pool_config['name']} stopped")
    
    def generate_trex_job_id(self):
        """Generate a T-Rex compatible job ID"""
        self.job_id_counter += 1
        return f"{self.job_id_counter:08x}"

    def handle_share_submission(self, message):
        """Submit share to all pools with timing optimization"""
        submission_start = time.time()
        
        # Use lock to prevent concurrent submissions that could cause timing issues
        with self.share_submission_lock:
            params = message.get('params', [])
            logger.info(f"Received share submission: {params}")
            
            # Debug T-Rex submission format
            if len(params) >= 5:
                logger.info(f"T-Rex format analysis:")
                logger.info(f"  [0] Worker: {params[0]}")
                logger.info(f"  [1] Job ID: {params[1]}")
                logger.info(f"  [2] Extranonce2: {params[2]} (len: {len(params[2])})")
                logger.info(f"  [3] Ntime: {params[3]} (len: {len(params[3])})")
                logger.info(f"  [4] Nonce: {params[4]} (len: {len(params[4])})")
            
            # Submit to all pools in parallel with reduced timeout
            results = []
            threads = []
            
            for pool in self.pools:
                thread = threading.Thread(
                    target=self.submit_to_pool, 
                    args=(pool, params, results)
                )
                thread.start()
                threads.append(thread)
            
            # Wait for all submissions with reduced timeout for better timing
            for thread in threads:
                thread.join(timeout=3)  # Reduced from 5 to 3 seconds
            
            # Return success if any pool accepted
            accepted = any(results)
            
            submission_time = time.time() - submission_start
            logger.info(f"Share submission completed in {submission_time:.3f}s")
            
            return {
                "id": message.get('id'),
                "result": accepted,
                "error": None if accepted else [20, "All pools rejected", None]
            }

    def try_different_nonce_extractions(self, pool_name, trex_extranonce2, trex_result, trex_mixhash):
        """Try different methods to extract nonce from T-Rex KawPOW data"""
        
        # Remove 0x prefix from all fields
        if trex_extranonce2.startswith('0x'):
            extranonce2_hex = trex_extranonce2[2:]
        else:
            extranonce2_hex = trex_extranonce2
            
        if trex_result.startswith('0x'):
            result_hex = trex_result[2:]
        else:
            result_hex = trex_result
            
        if trex_mixhash.startswith('0x'):
            mixhash_hex = trex_mixhash[2:]
        else:
            mixhash_hex = trex_mixhash
            
        # Pool-specific method preferences based on known successful methods
        pool_preferences = {
            '2Miners': [
                'result_start_off4',  # Known to work for 2Miners
                'result_start',
                'result_start_off8',
                'extranonce2_end',
                'extranonce2_start',
            ],
            'Nanopool': [
                'extranonce2_start',  # Known to work for Nanopool
                'extranonce2_start_rev',
                'extranonce2_mid',
                'result_start',
                'result_start_off4',
            ],
            'Ravenminer': [
                'extranonce2_end',  # Ravenminer uses short extranonce2
                'extranonce2_end_rev',
                'extranonce2_mid',
                'result_start',
                'result_start_off4',
            ],
            'WoolyPooly': [
                'extranonce2_end',  # WoolyPooly uses fixed extranonce
                'result_start',
                'result_start_off4',
                'extranonce2_start',
                'result_mid',
            ],
            'HeroMiners': [
                'extranonce2_end',  # HeroMiners similar to WoolyPooly
                'result_start',
                'result_start_off4',
                'extranonce2_start',
                'result_mid',
            ]
        }
        
        # Get all possible extraction methods
        all_methods = {}
        
        # Method 1: From extranonce2 (different positions)
        if len(extranonce2_hex) >= 16:
            all_methods.update({
                "extranonce2_start": extranonce2_hex[:8],
                "extranonce2_start_rev": extranonce2_hex[6:14],
                "extranonce2_mid": extranonce2_hex[4:12],
                "extranonce2_mid2": extranonce2_hex[2:10],
                "extranonce2_end": extranonce2_hex[-8:],
                "extranonce2_end_rev": extranonce2_hex[-10:-2],
            })
        
        # Method 2: From result hash (many more positions)
        if len(result_hex) >= 16:
            all_methods.update({
                "result_start": result_hex[:8],
                "result_start_off4": result_hex[4:12],
                "result_start_off8": result_hex[8:16],
                "result_quarter": result_hex[16:24],
                "result_mid": result_hex[28:36],
                "result_mid_off4": result_hex[32:40],
                "result_3quarter": result_hex[48:56],
                "result_end_off8": result_hex[-16:-8],
                "result_end": result_hex[-8:],
                "result_end_off4": result_hex[-12:-4],
            })
        
        # Method 3: From mixhash (T-Rex specific)
        if len(mixhash_hex) >= 16:
            all_methods.update({
                "mixhash_start": mixhash_hex[:8],
                "mixhash_end": mixhash_hex[-8:],
                "mixhash_mid": mixhash_hex[28:36],
            })
        
        # Get pool-specific preferences
        preferred_methods = pool_preferences.get(pool_name, [])
        
        # Create ordered list: preferred methods first, then remaining methods
        nonce_candidates = []
        
        # Add preferred methods first
        for method in preferred_methods:
            if method in all_methods:
                nonce_candidates.append((method, all_methods[method]))
        
        # Add remaining methods
        for method, value in all_methods.items():
            if method not in preferred_methods:
                nonce_candidates.append((method, value))
        
        return nonce_candidates

    def get_optimized_nonce_method(self, pool_name, nonce_candidates, debug_info):
        """Get the best nonce extraction method for a pool"""
        
        # If we already know what works, use it
        if debug_info['successful_nonce_method']:
            for method, nonce in nonce_candidates:
                if method == debug_info['successful_nonce_method']:
                    return method, nonce, f"Using known successful method"
        
        # If we haven't found a working method yet, try methods in order
        if 'last_tried_method_index' not in debug_info:
            debug_info['last_tried_method_index'] = 0
        
        # Try next method in sequence
        method_index = debug_info['last_tried_method_index'] % len(nonce_candidates)
        selected_method, selected_nonce = nonce_candidates[method_index]
        debug_info['last_tried_method_index'] = method_index + 1
        
        # Show progress
        progress_msg = f"Trying method {method_index + 1}/{len(nonce_candidates)}"
        if method_index < 5:  # First 5 methods are pool-specific preferences
            progress_msg += " (pool-optimized)"
        
        return selected_method, selected_nonce, progress_msg

    def submit_to_pool(self, pool_config, params, results):
        """Submit share to a specific pool with timing optimization"""
        pool_name = pool_config['name']
        submit_start = time.time()
        
        try:
            # Each pool has its own dedicated adapter that knows exactly how to format submissions
            from adapters.factory import AdapterFactory
            adapter = AdapterFactory.create_adapter(pool_config)
            adapter.connect()
            
            if adapter and adapter.connected:
                # Initialize pool debug info if not exists
                if pool_name not in self.pool_debug_info:
                    self.pool_debug_info[pool_name] = {
                        'submissions': 0,
                        'accepted': 0,
                        'rejected': 0,
                        'last_errors': [],
                        'successful_nonce_method': None
                    }
                
                debug_info = self.pool_debug_info[pool_name]
                debug_info['submissions'] += 1
                
                logger.info(f"[DEBUG] {pool_name} - Submission #{debug_info['submissions']}")
                
                # Parse T-Rex KawPOW submission format: [worker, job_id, extranonce2, mixhash, result]
                trex_worker = params[0]       # Base address from T-Rex
                trex_extranonce2 = params[2]  # Contains nonce info
                trex_mixhash = params[3]      # KawPOW mix hash (66 chars)
                trex_result = params[4]       # KawPOW result hash (66 chars)
                
                # Use the pool's configured user string instead of T-Rex's base address
                # This ensures worker names are properly included
                correct_user = pool_config['user']
                logger.info(f"[DEBUG] {pool_name} - Using configured user: {correct_user} (T-Rex sent: {trex_worker})")
                
                # Try different nonce extraction methods
                nonce_candidates = self.try_different_nonce_extractions(
                    pool_name, trex_extranonce2, trex_result, trex_mixhash
                )
                
                # Get optimized method selection
                selected_method, selected_nonce, selection_reason = self.get_optimized_nonce_method(
                    pool_name, nonce_candidates, debug_info
                )
                
                nonce = int(selected_nonce, 16)
                
                # Get method index for better debugging
                method_index = next((i for i, (method, _) in enumerate(nonce_candidates) if method == selected_method), -1)
                
                logger.info(f"[DEBUG] {pool_name} - Available nonce extraction methods: {len(nonce_candidates)}")
                logger.info(f"[DEBUG] {pool_name} - Preview: {[f'{method}:{nonce}' for method, nonce in nonce_candidates[:5]]}")
                logger.info(f"[DEBUG] {pool_name} - Using method '{selected_method}' (#{method_index+1}/{len(nonce_candidates)}): {selected_nonce} (decimal: {nonce})")
                logger.info(f"[DEBUG] {pool_name} - Selection reason: {selection_reason}")
                logger.info(f"[DEBUG] {pool_name} - Using T-Rex extranonce2: {trex_extranonce2}")
                
                # Resolve the T-Rex job ID to the original pool job ID
                trex_job_id = params[1]
                original_job_id = self.trex_to_pool_job_mapping.get(trex_job_id, trex_job_id)
                
                if original_job_id != trex_job_id:
                    logger.info(f"[DEBUG] {pool_name} - Mapped T-Rex job ID {trex_job_id} to original pool job ID {original_job_id}")
                
                # Use stored job data with stricter timing requirements
                stored_job = self.job_storage.get(original_job_id)
                
                if stored_job:
                    # Check if job is too old - use stricter limit
                    job_age = time.time() - stored_job['timestamp']
                    if job_age > self.job_age_limit:
                        logger.warning(f"[DEBUG] {pool_name} - Job {original_job_id} is {job_age:.1f}s old, may be stale")
                        
                        # Try to find a fresher job
                        fresh_job = self.find_fresh_job()
                        if fresh_job:
                            logger.info(f"[DEBUG] {pool_name} - Using fresher job {fresh_job['job_id']} (age: {time.time() - fresh_job['timestamp']:.1f}s)")
                            stored_job = fresh_job
                            original_job_id = fresh_job['job_id']
                    
                    # Use original job data with the original job ID
                    job = {
                        'job_id': original_job_id,  # Use original pool job ID
                        'header_hash': stored_job['header_hash'],
                        'target': stored_job['target'],
                        'ntime': stored_job['ntime'],
                        'trex_extranonce2': trex_extranonce2,
                        'timestamp': stored_job['timestamp']
                    }
                    logger.info(f"[DEBUG] {pool_name} - Using stored job data, ntime: {stored_job['ntime']}, age: {time.time() - stored_job['timestamp']:.1f}s")
                else:
                    # Fallback to minimal job data
                    current_time = int(time.time())
                    ntime_32bit = f"{current_time:08x}"
                    job = {
                        'job_id': original_job_id,
                        'ntime': ntime_32bit,
                        'trex_extranonce2': trex_extranonce2
                    }
                    logger.warning(f"[DEBUG] {pool_name} - No stored job data found, using fallback ntime: {ntime_32bit}")
                
                # The dedicated adapter handles all the pool-specific formatting!
                response = adapter.submit_share(nonce, job)
                success = isinstance(response, dict) and not response.get('error')
                
                submit_time = time.time() - submit_start
                
                if success:
                    debug_info['accepted'] += 1
                    # This method works! Save it for future use
                    debug_info['successful_nonce_method'] = selected_method
                    logger.info(f"[SUCCESS] {pool_name} - Share accepted! Method '{selected_method}' works! (Success rate: {debug_info['accepted']}/{debug_info['submissions']}) [Time: {submit_time:.3f}s]")
                    logger.info(f"[LEARN] {pool_name} - Learned successful method: '{selected_method}' will be used for future submissions")
                else:
                    debug_info['rejected'] += 1
                    error_msg = response.get('error', 'Unknown error')
                    debug_info['last_errors'].append(error_msg)
                    if len(debug_info['last_errors']) > 5:
                        debug_info['last_errors'] = debug_info['last_errors'][-5:]
                    
                    logger.warning(f"[REJECT] {pool_name} - Share rejected: {error_msg} [Time: {submit_time:.3f}s]")
                    logger.info(f"[STATS] {pool_name} - Stats: {debug_info['accepted']} accepted, {debug_info['rejected']} rejected")
                    logger.info(f"[DEBUG] {pool_name} - Job details: header={job['header_hash'][:16] if 'header_hash' in job else 'N/A'}..., target={job['target'][:16] if 'target' in job else 'N/A'}...")
                    
                    # Show what method will be tried next
                    if not debug_info['successful_nonce_method']:
                        next_method_index = debug_info.get('last_tried_method_index', 0) % len(nonce_candidates)
                        next_method = nonce_candidates[next_method_index][0]
                        next_type = "(pool-optimized)" if next_method_index < 5 else "(fallback)"
                        logger.info(f"[INFO] {pool_name} - Will try '{next_method}' method next (#{next_method_index+1}/{len(nonce_candidates)}) {next_type}")
                
                results.append(success)
                
            else:
                logger.error(f"[ERROR] Could not connect to {pool_name}")
                results.append(False)
        
        except Exception as e:
            logger.error(f"[ERROR] Error submitting to {pool_name}: {e}")
            results.append(False)

    def find_fresh_job(self):
        """Find the freshest job in storage"""
        if not self.job_storage:
            return None
        
        # Sort jobs by timestamp, newest first
        sorted_jobs = sorted(self.job_storage.items(), key=lambda x: x[1]['timestamp'], reverse=True)
        
        for job_id, job_data in sorted_jobs:
            age = time.time() - job_data['timestamp']
            if age <= self.job_age_limit:
                return job_data
        
        # If no fresh job found, return the newest one anyway
        return sorted_jobs[0][1] if sorted_jobs else None

    def forward_job_to_miner(self, job_message):
        """Forward real job data from pool to miner with timing optimization"""
        if hasattr(self, 'miner_socket') and self.miner_socket:
            try:
                current_time = time.time()
                
                # Rate limiting: don't forward jobs too frequently
                if current_time - self.last_job_time < self.job_update_frequency:
                    logger.debug(f"Skipping job update (too frequent, last: {current_time - self.last_job_time:.1f}s ago)")
                    return
                
                self.last_job_time = current_time
                
                # Store current job for share submission
                self.current_job = job_message
                
                # Get original pool job ID
                original_job_id = job_message['params'][0]
                
                # Generate T-Rex compatible job ID
                trex_job_id = self.generate_trex_job_id()
                
                # Create mappings
                self.trex_to_pool_job_mapping[trex_job_id] = original_job_id
                self.pool_to_trex_job_mapping[original_job_id] = trex_job_id
                
                # Store job data for later use in share submission
                job_params = job_message['params']
                self.job_storage[original_job_id] = {
                    'job_id': original_job_id,
                    'header_hash': job_params[1],
                    'target': job_params[3],
                    'ntime': job_params[6] if len(job_params) > 6 else job_params[5] if len(job_params) > 5 else "00000000",
                    'timestamp': current_time  # Use current time for better precision
                }
                
                # Clean up old job data more aggressively
                if len(self.job_storage) > self.max_stored_jobs:
                    # Remove oldest jobs by timestamp
                    sorted_jobs = sorted(self.job_storage.items(), key=lambda x: x[1]['timestamp'])
                    for old_job_id, _ in sorted_jobs[:-self.max_stored_jobs]:
                        del self.job_storage[old_job_id]
                        # Also clean up mappings
                        old_trex_id = self.pool_to_trex_job_mapping.pop(old_job_id, None)
                        if old_trex_id:
                            self.trex_to_pool_job_mapping.pop(old_trex_id, None)
                
                # Also clean up jobs that are too old
                old_jobs = [job_id for job_id, job_data in self.job_storage.items() 
                           if current_time - job_data['timestamp'] > self.job_age_limit * 2]
                for old_job_id in old_jobs:
                    del self.job_storage[old_job_id]
                    # Also clean up mappings
                    old_trex_id = self.pool_to_trex_job_mapping.pop(old_job_id, None)
                    if old_trex_id:
                        self.trex_to_pool_job_mapping.pop(old_trex_id, None)
                
                # Modify the job message to use T-Rex compatible job ID
                modified_job_message = job_message.copy()
                modified_job_message['params'] = list(job_message['params'])
                modified_job_message['params'][0] = trex_job_id
                
                # Forward the modified job message to T-Rex
                self.miner_socket.send((json.dumps(modified_job_message) + "\n").encode())
                logger.info(f"Forwarded job {original_job_id} to miner as {trex_job_id}")
                
            except Exception as e:
                logger.error(f"Error forwarding job to miner: {e}")
    
    def forward_to_miner(self, message):
        """Forward message to miner"""
        if hasattr(self, 'miner_socket') and self.miner_socket:
            try:
                self.miner_socket.send((json.dumps(message) + "\n").encode())
            except Exception as e:
                logger.error(f"Error forwarding message to miner: {e}")
    
    def send_job_to_miner(self):
        """Send initial job to miner"""
        # This could be enhanced to send real job data
        pass

    def print_pool_statistics(self):
        """Print detailed pool statistics"""
        logger.info("=" * 50)
        logger.info("[STATS] POOL STATISTICS (Enhanced Learning System)")
        logger.info("=" * 50)
        
        for pool_name, debug_info in self.pool_debug_info.items():
            success_rate = (debug_info['accepted'] / debug_info['submissions']) * 100 if debug_info['submissions'] > 0 else 0
            
            logger.info(f"{pool_name}:")
            logger.info(f"  [RATE] Success Rate: {success_rate:.1f}% ({debug_info['accepted']}/{debug_info['submissions']})")
            logger.info(f"  [OK] Accepted: {debug_info['accepted']}")
            logger.info(f"  [REJECT] Rejected: {debug_info['rejected']}")
            
            if debug_info['successful_nonce_method']:
                logger.info(f"  [LEARN] Successful Method: {debug_info['successful_nonce_method']}")
            else:
                tried_methods = debug_info.get('last_tried_method_index', 0)
                logger.info(f"  [PROGRESS] Learning Progress: Tried {tried_methods} different methods")
                logger.info(f"  [STATUS] Status: Still searching for working nonce extraction method")
            
            if debug_info['last_errors']:
                logger.info(f"  [ERROR] Recent Error: {debug_info['last_errors'][-1]}")
            
            logger.info("")

if __name__ == "__main__":
    proxy = PoolProxy()
    
    print("Multi-Pool Ravencoin Proxy")
    print("=" * 40)
    print(f"1. Start this proxy")
    print(f"2. Configure your KawPOW miner:")
    print(f"   t-rex.exe -a kawpow -o stratum+tcp://localhost:4444 -u YOUR_WALLET -w rig1")
    print(f"3. The proxy will distribute your shares to all pools!")
    print(f"4. Press Ctrl+C to stop gracefully")
    print()
    
    # Start a background thread to show stats every 5 minutes
    def show_stats_periodically():
        while proxy.running:
            time.sleep(300)  # 5 minutes
            if proxy.running and proxy.pool_debug_info:
                proxy.print_pool_statistics() # Changed from show_pool_stats to print_pool_statistics
    
    stats_thread = threading.Thread(target=show_stats_periodically)
    stats_thread.daemon = True
    stats_thread.start()
    
    try:
        proxy.start_proxy_server()
    except KeyboardInterrupt:
        # Signal handler will take care of cleanup
        pass
    except Exception as e:
        logger.error(f"Proxy error: {e}")
    finally:
        if proxy.pool_debug_info:
            proxy.print_pool_statistics() # Changed from show_pool_stats to print_pool_statistics
        logger.info("Proxy stopped") 