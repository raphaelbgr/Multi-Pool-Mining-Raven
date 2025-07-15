import json
import subprocess
import time
import logging
import re
from datetime import datetime
from submit_share import submit_share

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kawpow_miner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KawpowMultiPoolMiner:
    def __init__(self):
        self.config = self.load_config()
        # Start from a random nonce to avoid always testing the same ranges
        import random
        self.nonce_start = random.randint(0, 1000000)
        self.cycle_count = 0
        self.total_shares = 0
        self.accepted_shares = 0
        self.pool_stats = {}
        
        # Initialize pool statistics
        for i, pool in enumerate(self.config['pools']):
            self.pool_stats[i] = {
                'name': pool['name'],
                'shares_found': 0,
                'shares_accepted': 0,
                'multi_pool_hits': 0  # Count of nonces valid for multiple pools
            }
        
    def load_config(self):
        try:
            with open("config.json") as f:
                config = json.load(f)
            logger.info("[OK] Config loaded successfully")
            logger.info(f"[POOLS] Configured pools: {len(config['pools'])}")
            for i, pool in enumerate(config['pools']):
                logger.info(f"  {i}: {pool['name']} ({pool['host']}:{pool['port']})")
            return config
        except Exception as e:
            logger.error(f"[ERROR] Failed to load config: {str(e)}")
            raise

    def run_cycle(self):
        self.cycle_count += 1
        logger.info(f"[CYCLE] #{self.cycle_count} (Starting nonce: {self.nonce_start})")
        
        # Step 1: Get Jobs from ALL pools
        logger.info("[FETCH] Fetching jobs from all pools...")
        if not self.run_get_jobs():
            logger.warning("[WARN] Failed to get jobs, retrying in 5s")
            time.sleep(5)
            return
            
        # Step 2: Load Jobs
        try:
            with open("jobs.json") as f:
                jobs = json.load(f)
            active_pools = len([j for j in jobs if j.get('header_hash')])
            logger.info(f"[OK] Loaded {active_pools}/{len(self.config['pools'])} active pools")
            
            # Log details of each job with age info
            import os
            jobs_file_age = 0
            if os.path.exists("jobs.json"):
                jobs_file_age = time.time() - os.path.getmtime("jobs.json")
            
            logger.info(f"[AGE] Jobs file is {jobs_file_age:.1f} seconds old")
            
            for job in jobs:
                if job.get('header_hash'):
                    logger.info(f"  [JOB] Pool {job['pool_index']} ({job['pool_name']}): Job {job['job_id']}")
                    logger.debug(f"    Header: {job['header_hash'][:16]}...")
                    logger.debug(f"    Target: {job['target'][:16]}...")
                else:
                    logger.warning(f"  [WARN] Pool {job['pool_index']}: No job received")
                    
        except Exception as e:
            logger.error(f"[ERROR] Failed to load jobs: {str(e)}")
            return

        # Step 3: Run KAWPOW Multi-Target Miner
        logger.info("[TARGET] Starting KAWPOW multi-target miner...")
        logger.info(f"[POWER] Testing each nonce against ALL {active_pools} pool targets simultaneously!")
        output = self.run_kawpow_miner()
        if not output:
            logger.error("[ERROR] KAWPOW miner returned no output")
            return
            
        # Step 4: Process Multi-Pool Results
        logger.info("[PROCESS] Processing KAWPOW multi-pool results...")
        self.process_kawpow_results(output, jobs)
        
        # Step 5: Increment nonce range
        increment = self.config.get('nonce_increment', 262144) * 2  # Smaller increment for fresher jobs
        self.nonce_start += increment
        logger.info(f"[NEXT] Next nonce range: {self.nonce_start}")
        
        # Step 6: Log statistics
        self.log_statistics()

    def run_get_jobs(self):
        try:
            logger.debug("[FETCH] Running get_jobs.py...")
            result = subprocess.run(
                ["python", "get_jobs.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.error("[ERROR] get_jobs.py failed:")
                logger.error(result.stderr)
                return False
            logger.debug("[OK] get_jobs.py completed successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] get_jobs.py timed out after 30s")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in get_jobs: {str(e)}")
            return False

    def run_kawpow_miner(self):
        try:
            logger.debug(f"[LAUNCH] Running KAWPOW miner with nonce: {self.nonce_start}")
            result = subprocess.run(
                ["./kawpow_multi_target.exe", str(self.nonce_start)],
                capture_output=True,
                text=True,
                timeout=15  # Shorter timeout for fresh jobs - KAWPOW jobs expire quickly
            )
            if result.returncode != 0:
                logger.error("[ERROR] kawpow_multi_target.exe failed:")
                logger.error(result.stderr)
                return ""
            
            logger.debug("[OK] KAWPOW miner completed")
            logger.debug("Miner output:")
            logger.debug(result.stdout)
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] kawpow_multi_target.exe timed out after 15s")
            return ""
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in KAWPOW miner: {str(e)}")
            return ""

    def process_kawpow_results(self, output, jobs):
        valid_shares = 0
        multi_pool_nonces = {}  # Track nonces that work for multiple pools
        
        for line in output.splitlines():
            if "JACKPOT! Nonce" in line:
                # Parse the multi-pool result line
                logger.info(f"[JACKPOT] {line}")
                
                # Extract nonce and pools from line like: "JACKPOT! Nonce 123456 valid for pools: 0 1 2"
                try:
                    match = re.search(r'JACKPOT! Nonce (\d+) valid for pools: (.+)', line)
                    if match:
                        nonce = int(match.group(1))
                        pools_str = match.group(2)
                        pool_indices = [int(p.strip()) for p in pools_str.split()]
                        
                        logger.info(f"[KAWPOW] Found valid nonce {nonce} for pools: {pool_indices}")
                        self.total_shares += 1
                        
                        # Track multi-pool nonces
                        if nonce not in multi_pool_nonces:
                            multi_pool_nonces[nonce] = []
                        multi_pool_nonces[nonce].extend(pool_indices)
                        
                        # Submit to each valid pool
                        for pool_idx in pool_indices:
                            if pool_idx < len(jobs):
                                job = jobs[pool_idx]
                                if job and job.get('header_hash'):
                                    self.pool_stats[pool_idx]['shares_found'] += 1
                                    
                                    # Get fresh jobs right before submission to avoid stale job errors
                                    logger.info(f"[FRESH] Getting fresh job for {job['pool_name']} before submission...")
                                    fresh_job_success = self.run_get_jobs()
                                    
                                    if fresh_job_success:
                                        # Load fresh jobs
                                        try:
                                            with open("jobs.json") as f:
                                                fresh_jobs = json.load(f)
                                            fresh_job = next((j for j in fresh_jobs if j['pool_index'] == pool_idx), None)
                                            if fresh_job and fresh_job.get('header_hash'):
                                                logger.info(f"[FRESH] Using fresh job {fresh_job['job_id']} for {fresh_job['pool_name']}")
                                                job = fresh_job  # Use fresh job
                                            else:
                                                logger.warning(f"[FRESH] No fresh job available for pool {pool_idx}, using original")
                                        except Exception as e:
                                            logger.error(f"[ERROR] Failed to load fresh jobs: {str(e)}")
                                            continue
                                    
                                    # Submit share
                                    try:
                                        logger.info(f"[SUBMIT] Submitting KAWPOW share to {job['pool_name']}")
                                        success = submit_share(
                                            job['pool_index'],
                                            job['job_id'],
                                            job['extranonce2'],
                                            job['ntime'],
                                            nonce,
                                            job['pool_name']
                                        )
                                        
                                        if success:
                                            self.pool_stats[pool_idx]['shares_accepted'] += 1
                                            self.accepted_shares += 1
                                            logger.info(f"[ACCEPTED] KAWPOW share accepted by {job['pool_name']}")
                                        else:
                                            logger.warning(f"[REJECTED] KAWPOW share rejected by {job['pool_name']}")
                                            
                                    except Exception as e:
                                        logger.error(f"[ERROR] Failed to submit KAWPOW share to {job['pool_name']}: {str(e)}")
                                else:
                                    logger.warning(f"[WARN] No valid job for pool {pool_idx}")
                            else:
                                logger.warning(f"[WARN] Invalid pool index {pool_idx}")
                        
                        # Count multi-pool hits
                        if len(pool_indices) > 1:
                            for pool_idx in pool_indices:
                                if pool_idx < len(self.pool_stats):
                                    self.pool_stats[pool_idx]['multi_pool_hits'] += 1
                            logger.info(f"[MULTI] Nonce {nonce} valid for {len(pool_indices)} pools!")
                        
                except Exception as e:
                    logger.error(f"[ERROR] Failed to parse JACKPOT line: {str(e)}")
                    continue

    def log_statistics(self):
        logger.info("=" * 60)
        logger.info("KAWPOW MULTI-POOL MINER STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total cycles: {self.cycle_count}")
        logger.info(f"Total shares found: {self.total_shares}")
        logger.info(f"Total shares accepted: {self.accepted_shares}")
        if self.total_shares > 0:
            acceptance_rate = (self.accepted_shares / self.total_shares) * 100
            logger.info(f"Acceptance rate: {acceptance_rate:.2f}%")
        
        logger.info("\nPool Statistics:")
        for pool_idx, stats in self.pool_stats.items():
            pool_name = stats['name']
            shares_found = stats['shares_found']
            shares_accepted = stats['shares_accepted']
            multi_hits = stats['multi_pool_hits']
            
            acceptance_rate = 0
            if shares_found > 0:
                acceptance_rate = (shares_accepted / shares_found) * 100
            
            logger.info(f"  {pool_name}:")
            logger.info(f"    Shares found: {shares_found}")
            logger.info(f"    Shares accepted: {shares_accepted}")
            logger.info(f"    Acceptance rate: {acceptance_rate:.2f}%")
            logger.info(f"    Multi-pool hits: {multi_hits}")
        
        logger.info("=" * 60)

def main():
    logger.info("=== KAWPOW Multi-Pool Miner Starting ===")
    logger.info("Algorithm: KAWPOW (ProgPoW)")
    logger.info("Multi-pool optimization: ENABLED")
    logger.info("CUDA acceleration: ENABLED")
    
    miner = KawpowMultiPoolMiner()
    
    try:
        while True:
            miner.run_cycle()
            time.sleep(1)  # Brief pause between cycles
    except KeyboardInterrupt:
        logger.info("Shutting down KAWPOW multi-pool miner...")
        miner.log_statistics()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        miner.log_statistics()

if __name__ == "__main__":
    main() 