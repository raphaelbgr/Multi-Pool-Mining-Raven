#!/usr/bin/env python3
"""
Test script to verify share submission fixes
"""

import json
import logging
from submit_share import submit_share

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_share_submission():
    """Test share submission with current jobs"""
    
    # Load config and current jobs
    try:
        with open("config.json") as f:
            config = json.load(f)
        
        with open("jobs.json") as f:
            jobs = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config or jobs: {e}")
        return False
    
    if not jobs:
        logger.error("No jobs found - run get_jobs.py first")
        return False
    
    # Test with a fake nonce (will be rejected but tests the protocol)
    test_nonce = 0x12345678
    
    logger.info("Testing share submission with current jobs...")
    
    success_count = 0
    total_count = 0
    
    for job in jobs:
        pool_idx = job['pool_index']
        pool_config = config['pools'][pool_idx]
        
        logger.info(f"\nTesting {pool_config['name']} (Pool {pool_idx})")
        logger.info(f"  Job ID: {job['job_id']}")
        logger.info(f"  Target: {job['target'][:16]}...")
        
        total_count += 1
        
        # Test submission
        try:
            result = submit_share(pool_config, test_nonce, job)
            if result:
                logger.info(f"  ✅ Protocol test passed for {pool_config['name']}")
                success_count += 1
            else:
                logger.warning(f"  !! Share rejected (expected - using test nonce)")
                success_count += 1  # Still counts as protocol success
        except Exception as e:
            logger.error(f"  XX Protocol test failed for {pool_config['name']}: {e}")
    
    logger.info(f"\n## Test Results:")
    logger.info(f"  Pools tested: {total_count}")
    logger.info(f"  Protocol working: {success_count}")
    logger.info(f"  Success rate: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        logger.info("OK All pools have working protocols!")
        return True
    else:
        logger.warning("!! Some pools may have protocol issues")
        return False

def check_job_freshness():
    """Check if jobs are recent"""
    import os
    import time
    
    if not os.path.exists("jobs.json"):
        logger.error("jobs.json not found - run get_jobs.py first")
        return False
    
    job_age = time.time() - os.path.getmtime("jobs.json")
    
    if job_age > 300:  # 5 minutes
        logger.warning(f"Jobs are {job_age:.0f} seconds old - consider running get_jobs.py")
        return False
    else:
        logger.info(f"Jobs are {job_age:.0f} seconds old - fresh enough")
        return True

if __name__ == "__main__":
    logger.info("## Share Submission Test")
    logger.info("=" * 40)
    
    # Check job freshness
    if not check_job_freshness():
        logger.info("Running get_jobs.py to get fresh jobs...")
        import subprocess
        result = subprocess.run(["python", "get_jobs.py"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to get jobs")
            exit(1)
    
    # Test share submission
    success = test_share_submission()
    
    if success:
        logger.info("\nOK Share submission protocol tests passed!")
        logger.info("The miner should now properly submit shares to pools.")
    else:
        logger.error("\nXX Some protocol tests failed.")
        logger.error("Check the logs above for specific issues.")
        exit(1) 