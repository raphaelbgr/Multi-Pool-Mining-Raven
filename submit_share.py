from adapters import AdapterFactory
import json

def submit_share(pool_config, nonce, job):
    adapter = None
    try:
        adapter = AdapterFactory.create_adapter(pool_config)
        adapter.connect()
        
        if not adapter.connected:
            print(f"❌ Could not connect to {pool_config['name']} to submit share")
            return False
            
        response = adapter.submit_share(nonce, job)
        
        if isinstance(response, dict) and response.get('error'):
            print(f"❌ Share rejected by {pool_config['name']}: {response.get('error')}")
            return False
        
        print(f"✅ Share accepted by {pool_config['name']}")
        return True
        
    except Exception as e:
        print(f"❌ Error submitting to {pool_config['name']}: {str(e)}")
        return False
    finally:
        if adapter:
            adapter.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: submit_share.py <pool_index> <nonce> <job_json>")
        sys.exit(1)
        
    pool_index = int(sys.argv[1])
    nonce = int(sys.argv[2])
    job = json.loads(sys.argv[3])
    
    config = json.load(open("config.json"))
    pool_config = config['pools'][pool_index]
    
    submit_share(pool_config, nonce, job)