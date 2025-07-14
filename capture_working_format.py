#!/usr/bin/env python3
"""
Capture Working Format
Acts as a proxy between a real working miner and pools to capture exact formats
"""

import socket
import json
import threading
import time
from datetime import datetime

class MinerPoolProxy:
    def __init__(self, pool_config, local_port):
        self.pool_config = pool_config
        self.local_port = local_port
        self.captured_data = []
        
    def start_proxy(self):
        """Start proxy server"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('localhost', self.local_port))
        server.listen(1)
        
        print(f"[PROXY] Listening on port {self.local_port} for {self.pool_config['name']}")
        print(f"[PROXY] Configure your miner to connect to: localhost:{self.local_port}")
        print(f"[PROXY] We'll forward to: {self.pool_config['host']}:{self.pool_config['port']}")
        
        while True:
            try:
                miner_sock, addr = server.accept()
                print(f"[CONNECT] Miner connected from {addr}")
                
                # Connect to real pool
                pool_sock = socket.create_connection(
                    (self.pool_config['host'], self.pool_config['port']), 
                    timeout=10
                )
                print(f"[CONNECT] Connected to {self.pool_config['name']}")
                
                # Start bidirectional forwarding with capture
                miner_thread = threading.Thread(
                    target=self.forward_with_capture,
                    args=(miner_sock, pool_sock, "MINER->POOL", self.pool_config['name'])
                )
                pool_thread = threading.Thread(
                    target=self.forward_with_capture,
                    args=(pool_sock, miner_sock, "POOL->MINER", self.pool_config['name'])
                )
                
                miner_thread.start()
                pool_thread.start()
                
            except Exception as e:
                print(f"[ERROR] Proxy error: {e}")
    
    def forward_with_capture(self, source, dest, direction, pool_name):
        """Forward data while capturing it"""
        try:
            while True:
                data = source.recv(4096)
                if not data:
                    break
                
                # Forward the data
                dest.send(data)
                
                # Capture and analyze
                try:
                    text_data = data.decode('utf-8', errors='ignore')
                    self.analyze_and_capture(text_data, direction, pool_name)
                except:
                    pass
                    
        except Exception as e:
            print(f"[ERROR] Forward error ({direction}): {e}")
        finally:
            try:
                source.close()
                dest.close()
            except:
                pass
    
    def analyze_and_capture(self, data, direction, pool_name):
        """Analyze and capture interesting data"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for line in data.strip().split('\n'):
            if not line.strip():
                continue
                
            try:
                parsed = json.loads(line)
                
                # Capture mining.submit from miner
                if direction == "MINER->POOL" and parsed.get('method') == 'mining.submit':
                    print(f"\n[CAPTURE] {timestamp} - SHARE SUBMISSION to {pool_name}")
                    print(f"[CAPTURE] Direction: {direction}")
                    print(f"[CAPTURE] Raw JSON: {line}")
                    print(f"[CAPTURE] Parsed:")
                    print(f"  Method: {parsed['method']}")
                    print(f"  ID: {parsed.get('id')}")
                    
                    params = parsed.get('params', [])
                    if len(params) >= 5:
                        print(f"  Worker: {params[0]}")
                        print(f"  Job ID: {params[1]}")
                        print(f"  Extranonce2: {params[2]} (len: {len(params[2])})")
                        print(f"  Ntime: {params[3]} (len: {len(params[3])})")
                        print(f"  Nonce: {params[4]} (len: {len(params[4])})")
                        
                        # Analyze nonce format
                        nonce_str = params[4]
                        try:
                            if nonce_str.startswith('0x'):
                                nonce_val = int(nonce_str, 16)
                                print(f"  Nonce Analysis: Hex with prefix, value: {nonce_val}")
                            else:
                                nonce_val = int(nonce_str, 16)
                                print(f"  Nonce Analysis: Hex without prefix, value: {nonce_val}")
                                
                                # Check endianness
                                big_endian = f"{nonce_val:08x}"
                                little_endian = f"{(nonce_val & 0xFF):02x}{(nonce_val >> 8 & 0xFF):02x}{(nonce_val >> 16 & 0xFF):02x}{(nonce_val >> 24 & 0xFF):02x}"
                                
                                if nonce_str.lower() == big_endian:
                                    print(f"  Nonce Format: BIG ENDIAN")
                                elif nonce_str.lower() == little_endian:
                                    print(f"  Nonce Format: LITTLE ENDIAN")
                                else:
                                    print(f"  Nonce Format: UNKNOWN")
                                    
                        except ValueError:
                            print(f"  Nonce Analysis: Not hex format")
                    
                    # Store for later analysis
                    self.captured_data.append({
                        'timestamp': timestamp,
                        'pool': pool_name,
                        'direction': direction,
                        'method': parsed['method'],
                        'params': params,
                        'raw': line
                    })
                
                # Capture responses to submissions
                elif direction == "POOL->MINER" and parsed.get('id') and 'result' in parsed:
                    print(f"\n[RESPONSE] {timestamp} - POOL RESPONSE from {pool_name}")
                    print(f"[RESPONSE] ID: {parsed.get('id')}")
                    print(f"[RESPONSE] Result: {parsed.get('result')}")
                    print(f"[RESPONSE] Error: {parsed.get('error')}")
                    
                    # Store response
                    self.captured_data.append({
                        'timestamp': timestamp,
                        'pool': pool_name,
                        'direction': direction,
                        'type': 'response',
                        'data': parsed,
                        'raw': line
                    })
                
                # Capture subscribe responses (to get extra_nonce)
                elif direction == "POOL->MINER" and parsed.get('method') == 'mining.notify':
                    print(f"\n[JOB] {timestamp} - JOB from {pool_name}")
                    params = parsed.get('params', [])
                    if params:
                        print(f"[JOB] Job ID: {params[0]}")
                        print(f"[JOB] Params: {len(params)} parameters")
                
            except json.JSONDecodeError:
                # Not JSON, might be important raw data
                if direction == "MINER->POOL" and any(word in line.lower() for word in ['submit', 'mining']):
                    print(f"\n[RAW] {timestamp} - Non-JSON from miner: {line}")
                elif direction == "POOL->MINER" and any(word in line.lower() for word in ['result', 'error']):
                    print(f"\n[RAW] {timestamp} - Non-JSON from pool: {line}")
    
    def save_captured_data(self, filename):
        """Save captured data to file"""
        with open(filename, 'w') as f:
            json.dump(self.captured_data, f, indent=2)
        print(f"[SAVE] Captured data saved to {filename}")

def start_capture_for_pool(pool_config, port_offset):
    """Start capture proxy for a specific pool"""
    local_port = 5000 + port_offset
    proxy = MinerPoolProxy(pool_config, local_port)
    
    print(f"\n=== STARTING CAPTURE FOR {pool_config['name']} ===")
    print(f"Local port: {local_port}")
    print(f"Target: {pool_config['host']}:{pool_config['port']}")
    
    # Start proxy in thread
    proxy_thread = threading.Thread(target=proxy.start_proxy)
    proxy_thread.daemon = True
    proxy_thread.start()
    
    return proxy

def main():
    print("WORKING FORMAT CAPTURE TOOL")
    print("===========================")
    print()
    print("This tool creates proxy servers for each pool.")
    print("Configure your working miner to connect to these proxies.")
    print("We'll capture exactly what the working miner sends.")
    print()
    
    # Load pools
    with open("config.json") as f:
        config = json.load(f)
    
    proxies = []
    
    # Start proxy for each pool
    for i, pool_config in enumerate(config['pools']):
        proxy = start_capture_for_pool(pool_config, i)
        proxies.append(proxy)
        time.sleep(1)  # Stagger startup
    
    print(f"\n=== ALL PROXIES STARTED ===")
    print("Configure your working miner to use these endpoints:")
    
    for i, pool_config in enumerate(config['pools']):
        local_port = 5000 + i
        print(f"  {pool_config['name']}: localhost:{local_port}")
    
    print("\nExample T-Rex commands:")
    for i, pool_config in enumerate(config['pools']):
        local_port = 5000 + i
        print(f"  # {pool_config['name']}")
        print(f"  t-rex.exe -a kawpow -o stratum+tcp://localhost:{local_port} -u {pool_config['user']} -w test")
    
    print(f"\nPress Ctrl+C to stop and save captured data...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[STOP] Stopping capture...")
        
        # Save all captured data
        for i, proxy in enumerate(proxies):
            if proxy.captured_data:
                filename = f"captured_{config['pools'][i]['name'].lower()}.json"
                proxy.save_captured_data(filename)
        
        print(f"[COMPLETE] Capture complete!")

if __name__ == "__main__":
    main() 