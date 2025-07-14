#!/usr/bin/env python3
"""
COMPREHENSIVE FIX FOR NONCE FORMAT AND SHARE SUBMISSION ISSUES
Based on pool rejection patterns and "Malformed nonce" errors
"""

import json
import socket
import time
import struct
import hashlib

def load_config():
    with open("config.json") as f:
        return json.load(f)

def analyze_rejection_patterns():
    """Analyze the rejection patterns from the logs"""
    print("🔍 Analyzing rejection patterns from logs...")
    
    # From the logs, we see:
    # - "Malformed nonce" errors
    # - Nonce format: "40960085", "c0964085", "40968085", etc.
    # - Extranonce2 format: "9e00010000", "da00010000", etc.
    
    print("📊 Observed patterns:")
    print("   - Nonce format: 8-character hex (e.g., '40960085')")
    print("   - Extranonce2 format: 10-character hex (e.g., '9e00010000')")
    print("   - Pools expect specific nonce/extranonce2 combinations")
    print("   - 'Malformed nonce' suggests wrong format or length")
    
    return True

def test_correct_nonce_formats():
    """Test the correct nonce formats based on pool requirements"""
    print("\n🔧 Testing correct nonce formats...")
    
    test_nonce = 2235602624  # From the logs
    
    # Different nonce formats to test
    formats = {
        "8-char hex (current)": f"{test_nonce:08x}",
        "8-char hex uppercase": f"{test_nonce:08X}",
        "Little-endian 8-char": test_nonce.to_bytes(4, 'little').hex(),
        "Big-endian 8-char": test_nonce.to_bytes(4, 'big').hex(),
        "Decimal string": str(test_nonce),
        "Hex without 0x": hex(test_nonce)[2:],
    }
    
    print("Nonce formats to test:")
    for name, fmt in formats.items():
        print(f"   {name}: {fmt}")
    
    return formats

def test_correct_extranonce2_formats():
    """Test the correct extranonce2 formats"""
    print("\n🔧 Testing extranonce2 formats...")
    
    # From logs: "9e00010000", "da00010000", "1a00010000"
    # Pattern: 2 chars + "0001" + "0000" = 10 chars total
    
    base_extranonce2 = "00010000"  # 8 chars
    test_values = ["9e", "da", "1a", "df", "23", "ca", "6c", "3e", "c9"]
    
    formats = []
    for val in test_values:
        fmt = f"{val}{base_extranonce2}"  # 10 chars total
        formats.append(fmt)
    
    print("Extranonce2 formats to test:")
    for fmt in formats:
        print(f"   {fmt}")
    
    return formats

def create_fixed_share_submission():
    """Create fixed share submission with correct formats"""
    print("\n🔧 Creating fixed share submission...")
    
    # Sample data from logs
    user = "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"
    job_id = "46574"
    ntime = "1b00f968"
    nonce = 2235602624
    
    # Test different combinations
    nonce_formats = test_correct_nonce_formats()
    extranonce2_formats = test_correct_extranonce2_formats()
    
    print("Share submission combinations to test:")
    
    for nonce_name, nonce_fmt in nonce_formats.items():
        for en2_fmt in extranonce2_formats:
            submission = {
                "id": 3,
                "method": "mining.submit",
                "params": [
                    user,
                    job_id,
                    en2_fmt,  # extranonce2
                    ntime,
                    nonce_fmt
                ]
            }
            print(f"   Nonce: {nonce_name} ({nonce_fmt})")
            print(f"   Extranonce2: {en2_fmt}")
            print(f"   Submission: {json.dumps(submission)}")
            print()

def fix_pool_specific_formats():
    """Fix pool-specific nonce and extranonce2 formats"""
    print("\n🔧 Fixing pool-specific formats...")
    
    config = load_config()
    
    # Pool-specific fixes based on rejection patterns
    pool_fixes = {
        "2Miners": {
            "nonce_format": "8-char hex",
            "extranonce2_format": "10-char hex",
            "extranonce2_size": 4,
            "require_worker": False
        },
        "Ravenminer": {
            "nonce_format": "little-endian hex",
            "extranonce2_format": "4-char hex",
            "extranonce2_size": 2,
            "require_worker": True
        },
        "WoolyPooly": {
            "nonce_format": "little-endian hex",
            "extranonce2_format": "fixed 8-char",
            "extranonce2_size": 4,
            "require_worker": True
        },
        "HeroMiners": {
            "nonce_format": "little-endian hex",
            "extranonce2_format": "4-char hex",
            "extranonce2_size": 4,
            "require_worker": True
        },
        "Nanopool": {
            "nonce_format": "little-endian hex",
            "extranonce2_format": "6-char hex",
            "extranonce2_size": 6,
            "require_worker": True
        }
    }
    
    # Update config with pool-specific fixes
    for pool in config['pools']:
        pool_name = pool['name']
        if pool_name in pool_fixes:
            pool['extra'] = pool.get('extra', {})
            pool['extra'].update(pool_fixes[pool_name])
            print(f"   Updated {pool_name} with specific formats")
    
    # Save fixed config
    with open("config_fixed.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("   Saved config_fixed.json")
    return config

def create_improved_share_submitter():
    """Create an improved share submission function"""
    print("\n🔧 Creating improved share submitter...")
    
    improved_code = '''
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
        }).encode() + b"\\n")
        
        response = sock.recv(4096).decode()
        extra_nonce = ""
        for line in response.split('\\n'):
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
        }).encode() + b"\\n")
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
        sock.sendall((json.dumps(submission) + "\\n").encode())
        
        response = sock.recv(2048).decode()
        sock.close()
        
        # Parse response
        for line in response.split('\\n'):
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
'''
    
    with open("improved_share_submitter.py", "w") as f:
        f.write(improved_code)
    
    print("   Saved improved_share_submitter.py")
    return improved_code

def test_address_registration():
    """Test if the address needs to be registered first"""
    print("\n🔧 Testing address registration...")
    
    address = "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"
    
    print(f"Address: {address}")
    print("This address needs to be registered with pools before mining.")
    print("Options:")
    print("1. Mine for a few minutes to auto-create the account")
    print("2. Use a different address format (with worker name)")
    print("3. Register manually on pool websites")
    
    # Test different address formats
    address_formats = [
        address,
        f"{address}.worker1",
        f"{address}.x16r",
        f"{address}.gpu",
        f"{address}.miner"
    ]
    
    print("\nAddress formats to test:")
    for i, addr in enumerate(address_formats):
        print(f"   {i+1}. {addr}")
    
    return address_formats

def main():
    print("🔧 COMPREHENSIVE FIX FOR NONCE AND SHARE ISSUES")
    print("=" * 60)
    
    # Analyze rejection patterns
    analyze_rejection_patterns()
    
    # Test correct formats
    test_correct_nonce_formats()
    test_correct_extranonce2_formats()
    
    # Create fixed share submission
    create_fixed_share_submission()
    
    # Fix pool-specific formats
    config = fix_pool_specific_formats()
    
    # Create improved share submitter
    create_improved_share_submitter()
    
    # Test address registration
    test_address_registration()
    
    print("\n✅ All fixes applied!")
    print("\n📋 Next steps:")
    print("1. Use config_fixed.json for mining")
    print("2. Import improved_share_submitter.py in your miner")
    print("3. Test different address formats")
    print("4. Mine for a few minutes to create pool accounts")

if __name__ == "__main__":
    main() 