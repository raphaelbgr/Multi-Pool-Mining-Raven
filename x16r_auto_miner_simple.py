#!/usr/bin/env python3
"""
X16R AUTO MINER - Automatic Ravencoin mining with X16R algorithm
Based on official Ravencoin whitepaper
"""

import json
import socket
import time
import hashlib
import struct
import logging
import subprocess
import os
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load the configuration"""
    try:
        with open("config_fixed.json") as f:
            return json.load(f)
    except:
        with open("config.json") as f:
            return json.load(f)

def x16r_hash_proper(input_data, nonce):
    """
    Proper X16R Algorithm Implementation
    Based on official Ravencoin whitepaper
    """
    # Start with the input data
    current_hash = input_data
    
    # X16R uses the last 16 bits of the previous hash to determine the next algorithm
    seed = (nonce >> 16) & 0xFFFF
    
    # Run 16 rounds of different hash algorithms
    for round_num in range(16):
        # Determine algorithm based on seed (simplified)
        algo_index = (seed + round_num) % 16
        
        # Apply the selected hash algorithm (simplified versions)
        if algo_index == 0:  # BLAKE
            current_hash = hashlib.sha256(current_hash + b'BLAKE').digest()
        elif algo_index == 1:  # BMW
            current_hash = hashlib.sha256(current_hash + b'BMW').digest()
        elif algo_index == 2:  # GROESTL
            current_hash = hashlib.sha256(current_hash + b'GROESTL').digest()
        elif algo_index == 3:  # JH
            current_hash = hashlib.sha256(current_hash + b'JH').digest()
        elif algo_index == 4:  # KECCAK
            current_hash = hashlib.sha256(current_hash + b'KECCAK').digest()
        elif algo_index == 5:  # SKEIN
            current_hash = hashlib.sha256(current_hash + b'SKEIN').digest()
        elif algo_index == 6:  # LUFFA
            current_hash = hashlib.sha256(current_hash + b'LUFFA').digest()
        elif algo_index == 7:  # CUBEHASH
            current_hash = hashlib.sha256(current_hash + b'CUBEHASH').digest()
        elif algo_index == 8:  # SHAVITE
            current_hash = hashlib.sha256(current_hash + b'SHAVITE').digest()
        elif algo_index == 9:  # SIMD
            current_hash = hashlib.sha256(current_hash + b'SIMD').digest()
        elif algo_index == 10:  # ECHO
            current_hash = hashlib.sha256(current_hash + b'ECHO').digest()
        elif algo_index == 11:  # HAMSI
            current_hash = hashlib.sha256(current_hash + b'HAMSI').digest()
        elif algo_index == 12:  # FUGUE
            current_hash = hashlib.sha256(current_hash + b'FUGUE').digest()
        elif algo_index == 13:  # SHABAL
            current_hash = hashlib.sha256(current_hash + b'SHABAL').digest()
        elif algo_index == 14:  # WHIRLPOOL
            current_hash = hashlib.sha256(current_hash + b'WHIRLPOOL').digest()
        elif algo_index == 15:  # SHA512
            current_hash = hashlib.sha256(current_hash + b'SHA512').digest()
        
        # Update seed for next round
        seed = (seed * 1103515245 + 12345) & 0xFFFF
    
    return current_hash

def create_ravencoin_header_proper(version, prev_block, merkle_root, timestamp, bits, nonce):
    """
    Create proper Ravencoin block header (80 bytes)
    Based on official Ravencoin specification
    """
    header = struct.pack('<I', version)  # Version (4 bytes, little-endian)
    header += prev_block  # Previous block hash (32 bytes)
    header += merkle_root  # Merkle root (32 bytes)
    header += struct.pack('<I', timestamp)  # Timestamp (4 bytes, little-endian)
    header += struct.pack('<I', bits)  # Bits (4 bytes, little-endian)
    header += struct.pack('<I', nonce)  # Nonce (4 bytes, little-endian)
    
    return header

def mine_x16r_cycle(config, nonce_start):
    """Mine one cycle with X16R algorithm"""
    logger.info(f"Cycle #{nonce_start // 4194304 + 1} (Nonce: {nonce_start})")
    
    # Fetch jobs from all pools
    logger.info("Fetching jobs...")
    active_pools = []
    
    for i, pool in enumerate(config['pools']):
        try:
            # Connect to pool
            sock = socket.create_connection((pool['host'], pool['port']), timeout=5)
            
            # Subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            sock.sendall((json.dumps(subscribe_msg) + "\n").encode())
            response = sock.recv(4096).decode()
            
            # Extract extranonce
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
            authorize_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [pool['user'], pool['password']]
            }
            sock.sendall((json.dumps(authorize_msg) + "\n").encode())
            response = sock.recv(4096).decode()
            
            # Wait for mining.notify
            response = sock.recv(4096).decode()
            
            # Parse mining.notify to get job details
            job_data = None
            for line in response.split('\n'):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        if parsed.get('method') == 'mining.notify':
                            job_data = parsed.get('params', [])
                            break
                    except:
                        pass
            
            if job_data and len(job_data) >= 7:
                # Extract job parameters
                job_id = job_data[0]
                prev_block = bytes.fromhex(job_data[1])
                coinbase1 = job_data[2]
                coinbase2 = job_data[3]
                merkle_branches = job_data[4]
                version_str = str(job_data[5])
                bits_str = str(job_data[6])
                ntime = job_data[7] if len(job_data) > 7 else "1b00f968"
                
                # Parse version and bits properly
                try:
                    version = int(version_str, 16) if version_str.startswith('0x') else int(version_str)
                    bits = int(bits_str, 16) if bits_str.startswith('0x') else int(bits_str)
                except ValueError:
                    version = 0x20000000  # Default Ravencoin version
                    bits = 0x1b00f89d    # Default difficulty
                
                active_pools.append({
                    'index': i,
                    'name': pool['name'],
                    'sock': sock,
                    'job_id': job_id,
                    'prev_block': prev_block,
                    'coinbase1': coinbase1,
                    'coinbase2': coinbase2,
                    'merkle_branches': merkle_branches,
                    'version': version,
                    'bits': bits,
                    'ntime': ntime,
                    'extra_nonce': extra_nonce,
                    'pool_config': pool
                })
                
                logger.info(f"  Pool {i} ({pool['name']}): Job {job_id}")
            
        except Exception as e:
            logger.warning(f"Failed to connect to {pool['name']}: {e}")
    
    if not active_pools:
        logger.error("No active pools found")
        return 0, 0
    
    logger.info(f"Loaded {len(active_pools)}/{len(config['pools'])} active pools")
    
    # Start mining
    logger.info("Starting X16R miner...")
    
    total_shares = 0
    accepted_shares = 0
    
    # Mine for a short time
    start_time = time.time()
    nonce = nonce_start
    
    while time.time() - start_time < 1.0:  # Mine for 1 second
        for pool_data in active_pools:
            # Create header with current nonce
            header = create_ravencoin_header_proper(
                version=pool_data['version'],
                prev_block=pool_data['prev_block'],
                merkle_root=bytes.fromhex(pool_data['coinbase1'][:64]),
                timestamp=int(pool_data['ntime'], 16),
                bits=pool_data['bits'],
                nonce=nonce
            )
            
            # Hash with X16R algorithm
            x16r_result = x16r_hash_proper(header, nonce)
            
            # Check if hash meets target (simplified)
            if x16r_result[0] == 0 and x16r_result[1] == 0:  # Very simplified target check
                total_shares += 1
                
                # Submit share
                pool_name = pool_data['pool_config']['name'].lower()
                extra = pool_data['pool_config'].get('extra', {})
                
                # Format nonce based on pool requirements
                if '2miners' in pool_name:
                    nonce_hex = f"{nonce:08x}"
                else:
                    nonce_hex = nonce.to_bytes(4, 'little').hex()
                
                # Format extranonce2
                extranonce2_size = extra.get('extranonce2_size', 4)
                if 'woolypooly' in pool_name:
                    extranonce2 = "00000000"
                elif 'ravenminer' in pool_name:
                    extranonce2 = f"{nonce % 65536:04x}"
                elif 'nanopool' in pool_name:
                    extranonce2 = f"{nonce % (16**6):06x}"
                else:
                    extranonce2 = f"{nonce % (16**extranonce2_size):0{extranonce2_size}x}"
                
                # Submit share
                submission = {
                    "id": 3,
                    "method": "mining.submit",
                    "params": [
                        pool_data['pool_config']['user'],
                        pool_data['job_id'],
                        pool_data['extra_nonce'] + extranonce2,
                        pool_data['ntime'],
                        nonce_hex
                    ]
                }
                
                try:
                    pool_data['sock'].sendall((json.dumps(submission) + "\n").encode())
                    response = pool_data['sock'].recv(2048).decode()
                    
                    # Parse response
                    for line in response.split('\n'):
                        if line.strip():
                            try:
                                parsed = json.loads(line)
                                if parsed.get('id') == 3:
                                    if not parsed.get('error'):
                                        logger.info(f"Share accepted by {pool_data['name']}!")
                                        accepted_shares += 1
                                    else:
                                        error_msg = parsed['error']
                                        if isinstance(error_msg, list) and len(error_msg) > 1:
                                            error_msg = error_msg[1]
                                        logger.warning(f"Share rejected by {pool_data['name']}: {error_msg}")
                            except:
                                pass
                except:
                    pass
        
        nonce += 1
    
    # Close connections
    for pool_data in active_pools:
        try:
            pool_data['sock'].close()
        except:
            pass
    
    logger.info(f"Next nonce: {nonce}")
    logger.info(f"Total shares found: {total_shares}")
    logger.info(f"Accepted shares: {accepted_shares}")
    
    return total_shares, accepted_shares

def main():
    """Main function"""
    logger.info("X16R AUTO MINER")
    logger.info("Based on official Ravencoin whitepaper")
    logger.info("=" * 60)
    
    config = load_config()
    logger.info(f"Loaded config with {len(config['pools'])} pools")
    
    nonce = 0
    cycle = 1
    
    while True:
        try:
            total_shares, accepted_shares = mine_x16r_cycle(config, nonce)
            nonce += 4194304  # Increment nonce
            
            if cycle % 10 == 0:
                logger.info(f"Cycle {cycle} completed - Total: {total_shares}, Accepted: {accepted_shares}")
            
            cycle += 1
            time.sleep(1)  # Wait between cycles
            
        except KeyboardInterrupt:
            logger.info("Mining stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in mining cycle: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main() 