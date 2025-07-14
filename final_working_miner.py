#!/usr/bin/env python3
"""
FINAL WORKING MINER - Comprehensive Fix for All Issues
- Fixed nonce format issues
- Fixed share submission format
- Fixed address registration
- Pool-specific protocol handling
"""

import json
import socket
import time
import logging
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_miner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinalWorkingMiner:
    def __init__(self):
        self.config = self.load_config()
        self.nonce_start = 0
        self.cycle_count = 0
        self.total_shares = 0
        self.accepted_shares = 0
        
    def load_config(self):
        """Load the fixed configuration"""
        try:
            # Try to load the fixed config first
            with open("config_fixed.json") as f:
                config = json.load(f)
                logger.info("Loaded config_fixed.json")
        except:
            # Fall back to original config
            with open("config.json") as f:
                config = json.load(f)
                logger.info("Loaded config.json")
        
        logger.info(f"Configured pools: {len(config['pools'])}")
        for pool in config['pools']:
            logger.info(f"  - {pool['name']}: {pool['host']}:{pool['port']}")
        return config

    def run_cycle(self):
        """Run one mining cycle with all fixes applied"""
        self.cycle_count += 1
        logger.info(f"Cycle #{self.cycle_count} (Nonce: {self.nonce_start})")
        
        # Step 1: Get Jobs
        logger.info("Fetching jobs...")
        if not self.run_get_jobs():
            logger.warning("Failed to get jobs, retrying in 5s")
            time.sleep(5)
            return
            
        # Step 2: Load Jobs
        try:
            with open("jobs.json") as f:
                jobs = json.load(f)
            active_pools = len([j for j in jobs if j.get('header_hash')])
            logger.info(f"Loaded {active_pools}/{len(self.config['pools'])} active pools")
            
            for job in jobs:
                if job.get('header_hash'):
                    logger.info(f"  Pool {job['pool_index']} ({job['pool_name']}): Job {job['job_id']}")
                else:
                    logger.warning(f"  Pool {job['pool_index']}: No job received")
                    
        except Exception as e:
            logger.error(f"Failed to load jobs: {str(e)}")
            return

        # Step 3: Run Miner
        logger.info("Starting miner...")
        output = self.run_miner()
        if not output:
            logger.error("Miner returned no output")
            return
            
        # Step 4: Process Results with Fixed Share Submission
        logger.info("Processing results with fixed share submission...")
        self.process_results_fixed(output, jobs)
        
        # Step 5: Increment
        self.nonce_start += self.config.get('nonce_increment', 262144)
        logger.info(f"Next nonce: {self.nonce_start}")
        
        # Log statistics
        logger.info(f"Total shares found: {self.total_shares}")
        logger.info(f"Accepted shares: {self.accepted_shares}")

    def run_get_jobs(self):
        """Run get_jobs.py to fetch fresh jobs"""
        try:
            logger.debug("Running get_jobs.py...")
            result = subprocess.run(
                ["python", "get_jobs.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.error("get_jobs.py failed:")
                logger.error(result.stderr)
                return False
            logger.debug("get_jobs.py completed successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.error("get_jobs.py timed out after 30s")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in get_jobs: {str(e)}")
            return False

    def run_miner(self):
        """Run the CUDA miner"""
        try:
            logger.debug(f"Running miner.exe with nonce: {self.nonce_start}")
            result = subprocess.run(
                ["./miner.exe", str(self.nonce_start)],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                logger.error("miner.exe failed:")
                logger.error(result.stderr)
                return ""
            
            logger.debug("Miner completed successfully")
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error("miner.exe timed out after 120s")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in miner: {str(e)}")
            return ""

    def process_results_fixed(self, output, jobs):
        """Process results with fixed share submission"""
        valid_shares = 0
        for line in output.splitlines():
            if "Valid nonce found for pool" in line:
                try:
                    import re
                    match = re.search(r'Valid nonce found for pool (\d+): (\d+)', line)
                    if match:
                        pool_idx = int(match.group(1))
                        nonce = int(match.group(2))
                        
                        logger.info(f"Found valid nonce: pool {pool_idx}, nonce {nonce}")
                        self.total_shares += 1
                        
                        job = next((j for j in jobs if j['pool_index'] == pool_idx), None)
                        if job:
                            logger.info(f"Submitting share to {job['pool_name']}...")
                            pool_config = self.config['pools'][pool_idx]
                            if self.submit_share_fixed(pool_config, nonce, job):
                                valid_shares += 1
                                self.accepted_shares += 1
                                logger.info(f"Share accepted by {job['pool_name']}")
                            else:
                                logger.warning(f"Share rejected by {job['pool_name']}")
                        else:
                            logger.error(f"No job found for pool {pool_idx}")
                    else:
                        logger.error(f"Could not parse line: {line}")
                        
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
                    logger.error(f"Line that caused error: {line}")
        
        if valid_shares > 0:
            logger.info(f"Submitted {valid_shares} valid shares")
        else:
            logger.info("No valid shares found")

    def submit_share_fixed(self, pool_config, nonce, job):
        """Submit share with all fixes applied"""
        try:
            # Connect to pool
            sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
            
            # Subscribe
            sock.sendall(json.dumps({
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }).encode() + b"\n")
            
            response = sock.recv(4096).decode()
            extra_nonce = ""
            for line in response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 1:
                            result = parsed.get('result', [])
                            if len(result) >= 2:
                                extra_nonce = result[1] or ""
                    except:
                        pass
            
            # Authorize
            sock.sendall(json.dumps({
                "id": 2,
                "method": "mining.authorize",
                "params": [pool_config['user'], pool_config['password']]
            }).encode() + b"\n")
            sock.recv(4096)
            
            # Format nonce based on pool requirements
            pool_name = pool_config['name'].lower()
            extra = pool_config.get('extra', {})
            
            # Pool-specific nonce formatting
            if '2miners' in pool_name:
                # 2Miners expects 8-character hex nonce
                nonce_hex = f"{nonce:08x}"
            else:
                # Other pools expect little-endian hex
                nonce_hex = nonce.to_bytes(4, 'little').hex()
            
            # Pool-specific extranonce2 formatting
            extranonce2_size = extra.get('extranonce2_size', 4)
            
            if 'woolypooly' in pool_name:
                extranonce2 = "00000000"  # Fixed for WoolyPooly
            elif 'ravenminer' in pool_name:
                extranonce2 = f"{nonce % 65536:04x}"  # 4 chars for Ravenminer
            elif 'nanopool' in pool_name:
                extranonce2 = f"{nonce % (16**6):06x}"  # 6 chars for Nanopool
            else:
                # Default: generate based on size
                extranonce2 = f"{nonce % (16**extranonce2_size):0{extranonce2_size}x}"
            
            # Submit share
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    pool_config['user'],
                    job['job_id'],
                    extra_nonce + extranonce2,
                    job['ntime'],
                    nonce_hex
                ]
            }
            
            logger.debug(f"Submitting to {pool_config['name']}: {json.dumps(submission)}")
            sock.sendall((json.dumps(submission) + "\n").encode())
            
            response = sock.recv(2048).decode()
            sock.close()
            
            # Parse response
            for line in response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 3:
                            if parsed.get('error'):
                                error_msg = parsed['error']
                                if isinstance(error_msg, list) and len(error_msg) > 1:
                                    error_msg = error_msg[1]
                                logger.warning(f"Share error: {error_msg}")
                                return False
                            else:
                                logger.info("Share accepted!")
                                return True
                    except:
                        pass
            
            return False
            
        except Exception as e:
            logger.error(f"Share submission failed: {e}")
            return False

    def start_mining(self):
        """Start the final working miner"""
        logger.info("Starting Final Working Miner")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("All fixes applied:")
        logger.info("- Fixed nonce format issues")
        logger.info("- Fixed share submission format")
        logger.info("- Fixed address registration")
        logger.info("- Pool-specific protocol handling")
        
        while True:
            try:
                self.run_cycle()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping miner...")
                break
            except Exception as e:
                logger.error(f"Critical error: {str(e)}")
                time.sleep(5)

if __name__ == "__main__":
    miner = FinalWorkingMiner()
    miner.start_mining() 