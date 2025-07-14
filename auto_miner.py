import json
import subprocess
import time
import logging
from datetime import datetime
from submit_share import submit_share

# Configurar logging detalhado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnifiedMiner:
    def __init__(self):
        self.config = self.load_config()
        self.nonce_start = 0
        self.cycle_count = 0
        self.total_shares = 0
        self.accepted_shares = 0
        
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

        # Step 3: Run Miner
        logger.info("Starting miner...")
        output = self.run_miner()
        if not output:
            logger.error("Miner returned no output")
            return
            
        # Step 4: Process Results
        logger.info("Processing results...")
        self.process_results(output, jobs)
        
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

    def run_miner(self):
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
            
            logger.debug("Miner output:")
            logger.debug(result.stdout)
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error("miner.exe timed out after 120s")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in miner: {str(e)}")
            return ""

    def process_results(self, output, jobs):
        valid_shares = 0
        for line in output.splitlines():
            if "Valid nonce found for pool" in line:
                try:
                    # Parse: "Valid nonce found for pool 0: 1331185344"
                    # Extrair pool_idx e nonce usando regex ou parsing mais robusto
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
                            if submit_share(pool_config, nonce, job):
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

if __name__ == "__main__":
    logger.info("Starting Unified Miner")
    logger.info(f"Started at: {datetime.now()}")
    miner = UnifiedMiner()
    
    while True:
        try:
            miner.run_cycle()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping miner...")
            break
        except Exception as e:
            logger.error(f"Critical error: {str(e)}")
            time.sleep(5)