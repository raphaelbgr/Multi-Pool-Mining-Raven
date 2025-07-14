from adapters import AdapterFactory
import json
import logging

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('submit_share.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def submit_share(pool_config, nonce, job):
    adapter = None
    try:
        logger.info(f">> Submitting share to {pool_config['name']}")
        logger.debug(f"  Pool: {pool_config['host']}:{pool_config['port']}")
        logger.debug(f"  User: {pool_config['user']}")
        logger.debug(f"  Job ID: {job['job_id']}")
        logger.debug(f"  Nonce: {nonce}")
        logger.debug(f"  Header: {job['header_hash'][:16]}...")
        logger.debug(f"  Target: {job['target'][:16]}...")
        
        adapter = AdapterFactory.create_adapter(pool_config)
        logger.debug(f"Created adapter: {type(adapter).__name__}")
        
        logger.info(f"-> Connecting to {pool_config['name']}...")
        adapter.connect()
        
        if not adapter.connected:
            logger.error(f"XX Could not connect to {pool_config['name']} to submit share")
            return False
            
        logger.info(f"OK Connected to {pool_config['name']}, submitting share...")
        response = adapter.submit_share(nonce, job)
        
        logger.debug(f"Response from {pool_config['name']}: {response}")
        
        if isinstance(response, dict):
            if response.get('error'):
                logger.error(f"XX Share rejected by {pool_config['name']}: {response.get('error')}")
                return False
            elif response.get('result') is True:
                logger.info(f"OK Share accepted by {pool_config['name']}")
                return True
            elif response.get('result') is False:
                logger.warning(f"!! Share rejected by {pool_config['name']} (result: false)")
                return False
            else:
                logger.warning(f"!! Unexpected response format from {pool_config['name']}: {response}")
                return False
        else:
            logger.error(f"XX Invalid response type from {pool_config['name']}: {type(response)}")
            return False
        
    except Exception as e:
        logger.error(f"XX Error submitting to {pool_config['name']}: {str(e)}")
        return False
    finally:
        if adapter:
            logger.debug(f"-- Closing connection to {pool_config['name']}")
            adapter.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        logger.error("Usage: submit_share.py <pool_index> <nonce> <job_json>")
        sys.exit(1)
        
    pool_index = int(sys.argv[1])
    nonce = int(sys.argv[2])
    job = json.loads(sys.argv[3])
    
    logger.info(f"[START] Starting share submission")
    logger.info(f"  Pool index: {pool_index}")
    logger.info(f"  Nonce: {nonce}")
    logger.info(f"  Job: {job['job_id']}")
    
    config = json.load(open("config.json"))
    pool_config = config['pools'][pool_index]
    
    success = submit_share(pool_config, nonce, job)
    if success:
        logger.info("[SUCCESS] Share submission completed successfully")
    else:
        logger.error("[FAIL] Share submission failed")
        sys.exit(1)