import json
import subprocess
import time
from submit_share import submit_share

class UnifiedMiner:
    def __init__(self):
        self.config = self.load_config()
        self.nonce_start = 0
        self.cycle_count = 0
        
    def load_config(self):
        try:
            with open("config.json") as f:
                config = json.load(f)
            print("✅ Config loaded successfully")
            return config
        except Exception as e:
            print(f"❌ Failed to load config: {str(e)}")
            raise

    def run_cycle(self):
        self.cycle_count += 1
        print(f"\n🔁 Cycle #{self.cycle_count} (Nonce: {self.nonce_start})")
        
        # Step 1: Get Jobs
        print("🔄 Fetching jobs...")
        if not self.run_get_jobs():
            print("🛑 Failed to get jobs, retrying in 5s")
            time.sleep(5)
            return
            
        # Step 2: Load Jobs
        try:
            with open("jobs.json") as f:
                jobs = json.load(f)
            active_pools = len([j for j in jobs if j.get('header_hash')])
            print(f"📊 Loaded {active_pools}/{len(self.config['pools'])} active pools")
        except Exception as e:
            print(f"❌ Failed to load jobs: {str(e)}")
            return

        # Step 3: Run Miner
        print("⛏️ Starting miner...")
        output = self.run_miner()
        if not output:
            print("🛑 Miner returned no output")
            return
            
        # Step 4: Process Results
        print("📊 Processing results...")
        self.process_results(output, jobs)
        
        # Step 5: Increment
        self.nonce_start += self.config.get('nonce_increment', 262144)
        print(f"➡️ Next nonce: {self.nonce_start}")

    def run_get_jobs(self):
        try:
            result = subprocess.run(
                ["python", "get_jobs.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                print("❌ get_jobs.py failed:")
                print(result.stderr)
                return False
            return True
        except subprocess.TimeoutExpired:
            print("❌ get_jobs.py timed out after 30s")
            return False
        except Exception as e:
            print(f"❌ Unexpected error in get_jobs: {str(e)}")
            return False

    def run_miner(self):
        try:
            result = subprocess.run(
                ["./miner.exe", str(self.nonce_start)],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                print("❌ miner.exe failed:")
                print(result.stderr)
                return ""
            return result.stdout
        except subprocess.TimeoutExpired:
            print("❌ miner.exe timed out after 120s")
            return ""
        except Exception as e:
            print(f"❌ Unexpected error in miner: {str(e)}")
            return ""

    def process_results(self, output, jobs):
        valid_shares = 0
        for line in output.splitlines():
            if "Valid nonce found for pool" in line:
                try:
                    parts = line.strip().split()
                    pool_idx = int(parts[4])
                    nonce = int(parts[6])
                    
                    job = next((j for j in jobs if j['pool_index'] == pool_idx), None)
                    if job:
                        pool_config = self.config['pools'][pool_idx]
                        if submit_share(pool_config, nonce, job):
                            valid_shares += 1
                except Exception as e:
                    print(f"⚠️ Error processing result: {str(e)}")
        
        if valid_shares > 0:
            print(f"🎉 Submitted {valid_shares} valid shares")
        else:
            print("⚠️ No valid shares found")

if __name__ == "__main__":
    print("🚀 Starting Unified Miner")
    miner = UnifiedMiner()
    
    while True:
        miner.run_cycle()
        time.sleep(1)