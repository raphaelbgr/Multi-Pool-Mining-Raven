
def submit_share_improved(pool_config, nonce, job):
    """Improved share submission with correct formats"""
    try:
        import socket
        import json
        
        # Connect to pool
        sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
        
        # Subscribe
        sock.sendall(json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }).encode() + b"\n")
        
        response = sock.recv(4096).decode()
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
        sock.sendall(json.dumps({
            "id": 2,
            "method": "mining.authorize",
            "params": [pool_config['user'], pool_config['password']]
        }).encode() + b"\n")
        sock.recv(4096)
        
        # Format nonce based on pool requirements
        pool_name = pool_config['name'].lower()
        if '2miners' in pool_name:
            # 2Miners expects 8-character hex nonce
            nonce_hex = f"{nonce:08x}"
        else:
            # Other pools expect little-endian hex
            nonce_hex = nonce.to_bytes(4, 'little').hex()
        
        # Format extranonce2 based on pool requirements
        extra = pool_config.get('extra', {})
        extranonce2_size = extra.get('extranonce2_size', 4)
        
        if 'woolypooly' in pool_name:
            extranonce2 = "00000000"  # Fixed for WoolyPooly
        else:
            # Generate extranonce2 based on size
            extranonce2 = f"{nonce % (16**extranonce2_size):0{extranonce2_size}x}"
        
        # Submit share
        submission = {
            "id": 3,
            "method": "mining.submit",
            "params": [
                pool_config['user'],
                job['job_id'],
                extra_nonce + extranonce2,
                job['ntime'],
                nonce_hex
            ]
        }
        
        print(f"Submitting to {pool_config['name']}: {json.dumps(submission)}")
        sock.sendall((json.dumps(submission) + "\n").encode())
        
        response = sock.recv(2048).decode()
        sock.close()
        
        # Parse response
        for line in response.split('\n'):
            if line.strip():
                try:
                    parsed = json.loads(line)
                    if parsed.get('id') == 3:
                        if parsed.get('error'):
                            print(f"Share error: {parsed['error']}")
                            return False
                        else:
                            print("Share accepted!")
                            return True
                except:
                    pass
        
        return False
        
    except Exception as e:
        print(f"Share submission failed: {e}")
        return False
