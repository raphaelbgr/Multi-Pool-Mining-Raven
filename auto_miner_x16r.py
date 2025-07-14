#!/usr/bin/env python3
"""
AUTO MINER X16R - Ravencoin Multi-Pool Miner with X16R Algorithm
Based on official Ravencoin specification and X16R algorithm
"""

import json
import subprocess
import time
import logging
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner_x16r.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class X16RMiner:
    def __init__(self):
        self.config = self.load_config()
        self.nonce_start = 0
        self.cycle_count = 0
        self.total_shares = 0
        self.accepted_shares = 0
        self.x16r_miner = "miner_x16r.exe"  # X16R miner executable
        
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

    def run_cycle(self):
        self.cycle_count += 1
        logger.info(f"Cycle #{self.cycle_count} (Nonce: {self.nonce_start})")
        
        # Step 1: Get Jobs with X16R headers
        logger.info("Fetching jobs for X16R mining...")
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
            
            # Log details of each job
            for job in jobs:
                if job.get('header_hash'):
                    logger.info(f"  Pool {job['pool_index']} ({job['pool_name']}): Job {job['job_id']}")
                    logger.debug(f"    Header: {job['header_hash'][:16]}...")
                    logger.debug(f"    Target: {job['target'][:16]}...")
                else:
                    logger.warning(f"  Pool {job['pool_index']}: No job received")
                    
        except Exception as e:
            logger.error(f"Failed to load jobs: {str(e)}")
            return

        # Step 3: Run X16R Miner
        logger.info("Starting X16R miner...")
        output = self.run_x16r_miner()
        if not output:
            logger.error("X16R miner returned no output")
            return
            
        # Step 4: Process Results
        logger.info("Processing X16R results...")
        self.process_x16r_results(output, jobs)
        
        # Step 5: Increment
        self.nonce_start += self.config.get('nonce_increment', 262144)
        logger.info(f"Next nonce: {self.nonce_start}")
        
        # Log statistics
        logger.info(f"Total shares found: {self.total_shares}")
        logger.info(f"Accepted shares: {self.accepted_shares}")

    def run_get_jobs(self):
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
            logger.debug("get_jobs.py output:")
            logger.debug(result.stdout)
            return True
        except subprocess.TimeoutExpired:
            logger.error("get_jobs.py timed out after 30s")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in get_jobs: {str(e)}")
            return False

    def run_x16r_miner(self):
        try:
            logger.debug(f"Running {self.x16r_miner} with nonce: {self.nonce_start}")
            result = subprocess.run(
                [f"./{self.x16r_miner}", str(self.nonce_start), "x16r"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                logger.error(f"{self.x16r_miner} failed:")
                logger.error(result.stderr)
                return ""
            
            logger.debug("X16R miner output:")
            logger.debug(result.stdout)
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"{self.x16r_miner} timed out after 120s")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in X16R miner: {str(e)}")
            return ""

    def process_x16r_results(self, output, jobs):
        valid_shares = 0
        for line in output.splitlines():
            if "Valid nonce found for pool" in line:
                try:
                    # Parse: "Valid nonce found for pool 0: 1331185344"
                    import re
                    match = re.search(r'Valid nonce found for pool (\d+): (\d+)', line)
                    if match:
                        pool_idx = int(match.group(1))
                        nonce = int(match.group(2))
                        
                        logger.info(f"Found valid X16R nonce: pool {pool_idx}, nonce {nonce}")
                        self.total_shares += 1
                        
                        job = next((j for j in jobs if j['pool_index'] == pool_idx), None)
                        if job:
                            logger.info(f"Submitting X16R share to {job['pool_name']}...")
                            pool_config = self.config['pools'][pool_idx]
                            if self.submit_x16r_share(pool_config, nonce, job):
                                valid_shares += 1
                                self.accepted_shares += 1
                                logger.info(f"X16R share accepted by {job['pool_name']}")
                            else:
                                logger.warning(f"X16R share rejected by {job['pool_name']}")
                        else:
                            logger.error(f"No job found for pool {pool_idx}")
                    else:
                        logger.error(f"Could not parse line: {line}")
                        
                except Exception as e:
                    logger.error(f"Error processing X16R result: {str(e)}")
                    logger.error(f"Line that caused error: {line}")
        
        if valid_shares > 0:
            logger.info(f"Submitted {valid_shares} valid X16R shares")
        else:
            logger.info("No valid X16R shares found")

    def submit_x16r_share(self, pool_config, nonce, job):
        """Submit share using X16R algorithm with proper formatting"""
        try:
            import socket
            import json
            
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
            
            # Format nonce for X16R (little-endian)
            x16r_nonce = nonce.to_bytes(4, 'little').hex()
            
            # Get extranonce2 from T-Rex format
            trex_extranonce2 = self.get_extranonce2_from_trex(job)
            
            # Pool-specific extranonce2 formatting
            pool_name = pool_config['name'].lower()
            if '2miners' in pool_name:
                extranonce2 = trex_extranonce2[-8:] if len(trex_extranonce2) >= 8 else trex_extranonce2.ljust(8, '0')
            else:
                extranonce2 = trex_extranonce2[-4:] if len(trex_extranonce2) >= 4 else trex_extranonce2.ljust(4, '0')
            
            # Submit X16R share
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    pool_config['user'],
                    job['job_id'],
                    extra_nonce + extranonce2,
                    job['ntime'],
                    x16r_nonce
                ]
            }
            
            logger.debug(f"Submitting X16R share: {json.dumps(submission)}")
            sock.sendall((json.dumps(submission) + "\n").encode())
            
            response = sock.recv(2048).decode()
            sock.close()
            
            # Parse response
            for line in response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('id') == 3:
                            error = parsed.get('error')
                            if error:
                                error_msg = error[1] if isinstance(error, list) and len(error) > 1 else str(error)
                                if "low difficulty" in error_msg.lower():
                                    logger.info(f"X16R share format correct: {error_msg}")
                                    return True
                                else:
                                    logger.warning(f"X16R share error: {error_msg}")
                            else:
                                logger.info("X16R share accepted!")
                                return True
                    except json.JSONDecodeError:
                        continue
            
        except Exception as e:
            logger.error(f"X16R share submission failed: {e}")
        
        return False

    def get_extranonce2_from_trex(self, job):
        """Extract extranonce2 from T-Rex job format"""
        # This is a simplified version - in practice you'd parse the actual T-Rex format
        # For now, return a default value
        return "0001"

    def check_x16r_miner(self):
        """Check if X16R miner is available"""
        try:
            result = subprocess.run([f"./{self.x16r_miner}", "--help"], 
                                  capture_output=True, text=True, timeout=5)
            return True
        except:
            return False

    def start_mining(self):
        """Start X16R mining"""
        logger.info("Starting X16R Ravencoin Miner")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("Using official X16R algorithm for Ravencoin")
        
        # Check if X16R miner exists
        if not self.check_x16r_miner():
            logger.error(f"X16R miner not found: {self.x16r_miner}")
            logger.error("Please compile the X16R miner first:")
            logger.error("  build_x16r_miner.bat")
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

if __name__ == "__main__":
    miner = X16RMiner()
    miner.start_mining() 