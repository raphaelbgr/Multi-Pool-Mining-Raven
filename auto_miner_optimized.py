import json
import subprocess
import time
import logging
import re
from datetime import datetime
from submit_share import submit_share

# Configurar logging detalhado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedMultiPoolMiner:
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

        # Step 3: Run Optimized Multi-Target Miner
        logger.info("[TARGET] Starting OPTIMIZED multi-target miner...")
        logger.info(f"[POWER] Testing each nonce against ALL {active_pools} pool targets simultaneously!")
        output = self.run_optimized_miner()
        if not output:
            logger.error("[ERROR] Miner returned no output")
            return
            
        # Step 4: Process Multi-Pool Results
        logger.info("[PROCESS] Processing multi-pool results...")
        self.process_multi_pool_results(output, jobs)
        
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

    def run_optimized_miner(self):
        try:
            logger.debug(f"[LAUNCH] Running optimized miner with nonce: {self.nonce_start}")
            result = subprocess.run(
                ["./miner_multi_target.exe", str(self.nonce_start)],
                capture_output=True,
                text=True,
                timeout=15  # Shorter timeout for fresh jobs - KawPOW jobs expire quickly
            )
            if result.returncode != 0:
                logger.error("[ERROR] miner_multi_target.exe failed:")
                logger.error(result.stderr)
                return ""
            
            logger.debug("[OK] Optimized miner completed")
            logger.debug("Miner output:")
            logger.debug(result.stdout)
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] miner_multi_target.exe timed out after 15s")
            return ""
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in miner: {str(e)}")
            return ""

    def process_multi_pool_results(self, output, jobs):
        valid_shares = 0
        multi_pool_nonces = {}  # Track nonces that work for multiple pools
        
        for line in output.splitlines():
            if "JACKPOT! Nonce" in line:
                # Parse the multi-pool result line
                logger.info(f"[JACKPOT] {line}")
                
            elif "Valid nonce found for pool" in line:
                try:
                    # Parse: "Valid nonce found for pool 0: 1331185344"
                    match = re.search(r'Valid nonce found for pool (\d+): (\d+)', line)
                    if match:
                        pool_idx = int(match.group(1))
                        nonce = int(match.group(2))
                        
                        logger.info(f"[VALID] Found valid nonce: pool {pool_idx}, nonce {nonce}")
                        self.total_shares += 1
                        self.pool_stats[pool_idx]['shares_found'] += 1
                        
                        # Track multi-pool nonces
                        if nonce not in multi_pool_nonces:
                            multi_pool_nonces[nonce] = []
                        multi_pool_nonces[nonce].append(pool_idx)
                        
                        job = next((j for j in jobs if j['pool_index'] == pool_idx), None)
                        if job:
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
                                    logger.error(f"[FRESH] Error loading fresh jobs: {str(e)}")
                                    
                            logger.info(f"[SUBMIT] Submitting share to {job['pool_name']}...")
                            pool_config = self.config['pools'][pool_idx]
                            if submit_share(pool_config, nonce, job):
                                valid_shares += 1
                                self.accepted_shares += 1
                                self.pool_stats[pool_idx]['shares_accepted'] += 1
                                logger.info(f"[OK] Share accepted by {job['pool_name']}")
                            else:
                                logger.warning(f"[REJECT] Share rejected by {job['pool_name']}")
                        else:
                            logger.error(f"[ERROR] No job found for pool {pool_idx}")
                    else:
                        logger.error(f"[ERROR] Could not parse line: {line}")
                        
                except Exception as e:
                    logger.error(f"[ERROR] Error processing result: {str(e)}")
                    logger.error(f"Line that caused error: {line}")
        
        # Count multi-pool hits
        for nonce, pools in multi_pool_nonces.items():
            if len(pools) > 1:
                logger.info(f"[MULTI-HIT] Nonce {nonce} valid for {len(pools)} pools: {pools}")
                for pool_idx in pools:
                    self.pool_stats[pool_idx]['multi_pool_hits'] += 1
        
        if valid_shares > 0:
            logger.info(f"[STATS] Submitted {valid_shares} valid shares this cycle")
        else:
            logger.info("[STATS] No valid shares found this cycle")

    def log_statistics(self):
        logger.info("=" * 60)
        logger.info("[STATS] OPTIMIZED MULTI-POOL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"[CYCLES] Cycles completed: {self.cycle_count}")
        logger.info(f"[TOTAL] Total shares found: {self.total_shares}")
        logger.info(f"[ACCEPTED] Total shares accepted: {self.accepted_shares}")
        
        if self.total_shares > 0:
            success_rate = (self.accepted_shares / self.total_shares) * 100
            logger.info(f"[SUCCESS] Overall success rate: {success_rate:.1f}%")
        
        logger.info("\n[POOL-STATS] PER-POOL STATISTICS:")
        for pool_idx, stats in self.pool_stats.items():
            if stats['shares_found'] > 0:
                pool_success_rate = (stats['shares_accepted'] / stats['shares_found']) * 100
                logger.info(f"  {stats['name']}:")
                logger.info(f"    [FOUND] Shares found: {stats['shares_found']}")
                logger.info(f"    [ACCEPTED] Shares accepted: {stats['shares_accepted']} ({pool_success_rate:.1f}%)")
                logger.info(f"    [MULTI] Multi-pool hits: {stats['multi_pool_hits']}")
        
        logger.info("=" * 60)

if __name__ == "__main__":
    logger.info("[LAUNCH] Starting OPTIMIZED Multi-Pool Miner")
    logger.info(f"[TIME] Started at: {datetime.now()}")
    logger.info("[TARGET] This miner tests each nonce against ALL pool targets simultaneously!")
    
    miner = OptimizedMultiPoolMiner()
    
    while True:
        try:
            miner.run_cycle()
            time.sleep(0.5)  # Very short pause for fresh jobs
        except KeyboardInterrupt:
            logger.info("[STOP] Stopping miner...")
            miner.log_statistics()
            break
        except Exception as e:
            logger.error(f"[CRITICAL] Critical error: {str(e)}")
            time.sleep(5) 