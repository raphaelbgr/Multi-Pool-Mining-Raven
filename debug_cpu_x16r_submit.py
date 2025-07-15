import socket
import json
import struct
import hashlib
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pool connection info
HOST = "rvn.2miners.com"
PORT = 6060
USER = "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"
PASSWORD = "x"
WORKER = "cpu-debug"

# --- Helper functions ---
def double_sha256(data):
    """Double SHA256 (not X16R, just for format diagnostics!)"""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def mine_nonce(header, target_hex, start_nonce=0, max_tries=100000):
    """Try nonces until a hash is below the target (diagnostic only)"""
    target = int(target_hex, 16)
    for nonce in range(start_nonce, start_nonce + max_tries):
        header_with_nonce = header[:-4] + struct.pack('<I', nonce)
        hash_val = double_sha256(header_with_nonce)
        hash_int = int.from_bytes(hash_val[::-1], 'big')
        if hash_int < target:
            logging.info(f"Found valid nonce: {nonce:08x} (hash: {hash_val.hex()})")
            return nonce
    logging.warning("No valid nonce found in range.")
    return None

def build_header(job, extranonce2, ntime, nonce):
    """Builds a fake Ravencoin block header (diagnostic only)"""
    version = struct.pack('<I', job['version'])
    prev_hash = bytes.fromhex(job['prev_hash'])[::-1]
    merkle_root = bytes.fromhex(job['coinb1'])  # Not real, just for demo
    ntime_bytes = struct.pack('<I', int(ntime, 16))
    nbits_bytes = bytes.fromhex(job['nbits'])[::-1]
    nonce_bytes = struct.pack('<I', nonce)
    # Compose header (not real X16R, just for format)
    header = version + prev_hash + merkle_root + ntime_bytes + nbits_bytes + nonce_bytes
    # Pad to 80 bytes if needed
    if len(header) < 80:
        header += b'\x00' * (80 - len(header))
    return header[:80]

# --- Main debug logic ---
def main():
    logging.info("Connecting to 2Miners for debug job...")
    s = socket.create_connection((HOST, PORT), timeout=10)
    f = s.makefile("rw")

    # Subscribe
    subscribe = {"id": 1, "method": "mining.subscribe", "params": ["cpu-debug/0.1"]}
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
                            "ntime": hex(int(time.time()))[2:]  # Use current time as ntime
                        }
                        logging.info(f"Got job: {job_data}")
                        break
            except Exception as e:
                logging.warning(f"Failed to parse job: {e}")

    if not job_data:
        logging.error("No job received, aborting.")
        s.close()
        return

    # Build header and mine nonce (diagnostic only)
    extranonce2 = "00000000"
    ntime = job_data["ntime"]
    fake_header = build_header(job_data, extranonce2, ntime, 0)
    logging.info(f"Fake header (hex): {fake_header.hex()}")
    found_nonce = mine_nonce(fake_header, "00000000ffff0000000000000000000000000000000000000000000000000000", 0, 100000)

    if found_nonce is not None:
        # Submit share
        submit_msg = {
            "id": 88181504,
            "method": "mining.submit",
            "params": [
                job_data["job_id"],
                extranonce2,
                ntime,
                f"{found_nonce:08x}"
            ],
            "worker": WORKER
        }
        s.send((json.dumps(submit_msg) + "\n").encode())
        logging.info(f"Submitted share: {json.dumps(submit_msg)}")
        resp = f.readline()
        logging.info(f"Share submission response: {resp.strip()}")
    else:
        logging.warning("No valid nonce found, not submitting.")

    s.close()
    logging.info("Debug test complete.")

if __name__ == "__main__":
    print("\n[WARNING] This test uses double SHA256, not real X16R! It is for format/diagnostic only.\n")
    main() 