import socket
import json
import struct
import subprocess
import time
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pool connection info
HOST = "rvn.2miners.com"
PORT = 6060
USER = "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"
PASSWORD = "x"
WORKER = "cuda-debug"

# CUDA miner info
MINER_EXE = "miner_x16r_proper.exe"
POOL_DATA_SIZE = 4 + 80 + 32  # header_len + header + target

def create_headers_bin(job_data):
    """Create headers.bin file in the format expected by the CUDA miner"""
    try:
        # Build the 80-byte header
        version = struct.pack('<I', job_data['version'])
        prev_hash = bytes.fromhex(job_data['prev_hash'])[::-1]  # Reverse for little-endian
        merkle_root = bytes.fromhex(job_data['coinb1'])[:32]  # Use coinb1 as merkle root (simplified)
        ntime = struct.pack('<I', int(job_data['ntime'], 16))
        nbits = bytes.fromhex(job_data['nbits'])[::-1]  # Reverse for little-endian
        nonce = struct.pack('<I', 0)  # Start with nonce 0
        
        # Compose 80-byte header
        header = version + prev_hash + merkle_root + ntime + nbits + nonce
        
        # Target (difficulty 1 for Ravencoin)
        target = bytes.fromhex("00000000ffff0000000000000000000000000000000000000000000000000000")[::-1]
        
        # Write to headers.bin in the format expected by the CUDA miner
        with open("headers.bin", "wb") as f:
            # Write header length (4 bytes)
            f.write(struct.pack('<I', 80))
            # Write header (80 bytes)
            f.write(header)
            # Write target (32 bytes)
            f.write(target)
        
        logging.info(f"Created headers.bin with {len(header)} byte header")
        logging.info(f"Header hex: {header.hex()}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating headers.bin: {e}")
        return False

def run_cuda_miner(start_nonce=0, max_wait_time=300):
    """Run the CUDA miner and wait for a valid nonce"""
    try:
        logging.info(f"Running CUDA miner with start_nonce={start_nonce}")
        logging.info(f"Will run for up to {max_wait_time} seconds to find a valid nonce...")
        
        # Start the miner process
        process = subprocess.Popen(
            [MINER_EXE, str(start_nonce)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        found_nonce = None
        start_time = time.time()
        
        logging.info("🎯 CUDA miner is running... waiting for valid nonce...")
        
        while time.time() - start_time < max_wait_time:
            # Check if process is still running
            if process.poll() is not None:
                logging.warning("CUDA miner process ended unexpectedly")
                break
                
            # Read output line by line
            line = process.stdout.readline()
            if line:
                line = line.strip()
                logging.info(f"🔍 Miner: {line}")
                
                # Check for valid nonce
                if "Valid X16R nonce found for pool" in line:
                    try:
                        nonce_str = line.split(': ')[-1].strip()
                        found_nonce = int(nonce_str)
                        logging.info(f"🎉 FOUND VALID NONCE: {found_nonce}")
                        
                        # Kill the miner process since we found a nonce
                        process.terminate()
                        break
                    except Exception as e:
                        logging.error(f"Error parsing nonce from line: {e}")
            
            time.sleep(0.1)  # Small delay to avoid busy waiting
        
        # If process is still running, terminate it
        if process.poll() is None:
            process.terminate()
            logging.warning("⏰ CUDA miner timed out without finding valid nonce")
        
        if found_nonce is None:
            logging.warning("❌ No valid nonce found in this run")
            return None
            
        return found_nonce
        
    except Exception as e:
        logging.error(f"Error running CUDA miner: {e}")
        return None

def main():
    logging.info("Connecting to 2Miners for real job...")
    s = socket.create_connection((HOST, PORT), timeout=10)
    f = s.makefile("rw")

    # Subscribe
    subscribe = {"id": 1, "method": "mining.subscribe", "params": ["cuda-debug/0.1"]}
    s.send((json.dumps(subscribe) + "\n").encode())
    resp = f.readline()
    logging.info(f"Subscribe response: {resp.strip()}")

    # Authorize
    authorize = {"id": 2, "method": "mining.authorize", "params": [USER, PASSWORD], "worker": WORKER}
    s.send((json.dumps(authorize) + "\n").encode())
    resp = f.readline()
    logging.info(f"Authorize response: {resp.strip()}")

    # Wait for job
    job_data = None
    for _ in range(5):
        line = f.readline()
        if not line:
            break
        logging.info(f"Pool message: {line.strip()}")
        if "mining.notify" in line:
            try:
                data = json.loads(line)
                if data.get("method") == "mining.notify":
                    params = data.get("params", [])
                    if len(params) >= 7:
                        job_data = {
                            "job_id": params[0],
                            "prev_hash": params[1],
                            "coinb1": params[2],
                            "coinb2": params[3],
                            "version": params[5],
                            "nbits": params[6],
                            "ntime": hex(int(time.time()))[2:]  # Use current time
                        }
                        logging.info(f"Got real job: {job_data}")
                        break
            except Exception as e:
                logging.warning(f"Failed to parse job: {e}")

    if not job_data:
        logging.error("No job received, aborting.")
        s.close()
        return

    # Create headers.bin for the CUDA miner
    if not create_headers_bin(job_data):
        logging.error("Failed to create headers.bin, aborting.")
        s.close()
        return

    # Check if CUDA miner exists
    if not os.path.exists(MINER_EXE):
        logging.error(f"CUDA miner {MINER_EXE} not found!")
        logging.info("Please build the miner using: build_x16r_proper.bat")
        s.close()
        return

    # Run CUDA miner to find a valid nonce (up to 5 minutes)
    found_nonce = run_cuda_miner(0, 300)  # 5 minutes max

    if found_nonce is not None:
        logging.info(f"🚀 Submitting found nonce {found_nonce} to 2Miners...")
        
        # Submit the found nonce to the pool
        submit_msg = {
            "id": 88181504,
            "method": "mining.submit",
            "params": [
                job_data["job_id"],
                "00000000",  # extranonce2
                job_data["ntime"],
                f"{found_nonce:08x}"  # nonce in hex
            ],
            "worker": WORKER
        }
        
        logging.info(f"📤 Submit message: {json.dumps(submit_msg, indent=2)}")
        s.send((json.dumps(submit_msg) + "\n").encode())
        
        # Wait for response
        resp = f.readline()
        logging.info(f"📡 Pool response: {resp.strip()}")
        
        # Parse the response
        try:
            response_json = json.loads(resp)
            if response_json.get("result") == True:
                logging.info("✅ SHARE ACCEPTED! 🎉")
            elif response_json.get("error"):
                logging.error(f"❌ Share rejected: {response_json['error']}")
            else:
                logging.warning(f"⚠️ Unexpected response: {response_json}")
        except json.JSONDecodeError:
            logging.warning(f"⚠️ Could not parse response as JSON: {resp}")
            
    else:
        logging.warning("❌ No valid nonce found by CUDA miner, not submitting.")

    s.close()
    logging.info("Real X16R mining test complete.")

if __name__ == "__main__":
    print("\n[INFO] This test uses your real CUDA X16R miner to find valid shares!\n")
    main() 