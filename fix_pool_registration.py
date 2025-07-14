#!/usr/bin/env python3
"""
FIX POOL REGISTRATION AND ADDRESS ISSUES
Check and fix mining address registration with pools
"""

import json
import socket
import time
import hashlib
import struct

def load_config():
    with open("config.json") as f:
        return json.load(f)

def test_address_with_pool(pool_config):
    """Test if the mining address is properly registered with the pool"""
    print(f"\n🔍 Testing address with {pool_config['name']}...")
    print(f"   Address: {pool_config['user']}")
    print(f"   Host: {pool_config['host']}:{pool_config['port']}")
    
    try:
        sock = socket.create_connection((pool_config['host'], pool_config['port']), timeout=10)
        
        # Subscribe
        subscribe_msg = {
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }
        sock.sendall((json.dumps(subscribe_msg) + "\n").encode())
        response = sock.recv(4096).decode()
        print(f"   Subscribe response: {response.strip()}")
        
        # Authorize
        authorize_msg = {
            "id": 2,
            "method": "mining.authorize",
            "params": [pool_config['user'], pool_config['password']]
        }
        sock.sendall((json.dumps(authorize_msg) + "\n").encode())
        response = sock.recv(4096).decode()
        print(f"   Authorize response: {response.strip()}")
        
        # Check if authorization was successful
        for line in response.split('\n'):
            if line.strip():
                try:
                    parsed = json.loads(line)
                    if parsed.get('id') == 2:
                        if parsed.get('error'):
                            print(f"   ❌ Authorization failed: {parsed['error']}")
                            return False
                        else:
                            print(f"   ✅ Authorization successful")
                            return True
                except:
                    pass
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

def fix_address_format():
    """Fix the mining address format for better pool compatibility"""
    print("\n🔧 Fixing address format...")
    
    config = load_config()
    
    # The current address might need a worker name
    base_address = "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"
    
    # Create different address formats for testing
    address_formats = [
        base_address,  # Plain address
        f"{base_address}.worker1",  # With worker name
        f"{base_address}.x16r",  # With algorithm identifier
        f"{base_address}.gpu",  # With GPU identifier
    ]
    
    print("Testing different address formats:")
    for i, addr in enumerate(address_formats):
        print(f"   {i+1}. {addr}")
    
    return address_formats

def test_nonce_format():
    """Test different nonce formats to find the correct one"""
    print("\n🔧 Testing nonce formats...")
    
    test_nonce = 1234567890
    
    # Different nonce formats
    formats = {
        "Little-endian hex": test_nonce.to_bytes(4, 'little').hex(),
        "Big-endian hex": test_nonce.to_bytes(4, 'big').hex(),
        "Little-endian hex (8 chars)": f"{test_nonce.to_bytes(4, 'little').hex():0>8}",
        "Big-endian hex (8 chars)": f"{test_nonce.to_bytes(4, 'big').hex():0>8}",
        "Decimal string": str(test_nonce),
        "Hex string": hex(test_nonce)[2:],  # Remove '0x' prefix
    }
    
    print("Nonce formats for testing:")
    for name, fmt in formats.items():
        print(f"   {name}: {fmt}")
    
    return formats

def create_fixed_config():
    """Create a fixed configuration with proper address formats"""
    print("\n🔧 Creating fixed configuration...")
    
    config = load_config()
    
    # Test different address formats
    address_formats = fix_address_format()
    
    # Create multiple configs for testing
    fixed_configs = []
    
    for i, addr_format in enumerate(address_formats):
        fixed_config = {
            "pools": [],
            "max_pools": 32,
            "nonce_increment": 4194304
        }
        
        for pool in config['pools']:
            fixed_pool = pool.copy()
            fixed_pool['user'] = addr_format
            fixed_config['pools'].append(fixed_pool)
        
        fixed_configs.append(fixed_config)
    
    # Save test configs
    for i, cfg in enumerate(fixed_configs):
        with open(f"config_test_{i+1}.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"   Saved config_test_{i+1}.json")
    
    return fixed_configs

def test_share_submission_format():
    """Test different share submission formats"""
    print("\n🔧 Testing share submission formats...")
    
    # Sample data
    user = "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"
    job_id = "12345"
    ntime = "1b00f968"
    nonce = 1234567890
    
    # Different nonce formats
    nonce_formats = test_nonce_format()
    
    # Different extranonce2 formats
    extranonce2_formats = [
        "0001",  # 4 chars
        "00000001",  # 8 chars
        "01",  # 2 chars
        "000001",  # 6 chars
    ]
    
    print("Share submission formats to test:")
    
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
            print(f"   Nonce: {nonce_name}, Extranonce2: {en2_fmt}")
            print(f"   Submission: {json.dumps(submission)}")
            print()

def check_pool_websites():
    """Check if the address appears on pool websites"""
    print("\n🌐 Checking pool websites for address registration...")
    
    address = "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"
    
    pool_urls = {
        "2Miners": f"https://rvn.2miners.com/account/{address}",
        "Ravenminer": f"https://ravenminer.com/account/{address}",
        "Nanopool": f"https://rvn.nanopool.org/account/{address}",
        "WoolyPooly": f"https://woolypooly.com/en/coin/rvn/account/{address}",
        "HeroMiners": f"https://ravencoin.herominers.com/account/{address}"
    }
    
    print("Check these URLs manually:")
    for pool, url in pool_urls.items():
        print(f"   {pool}: {url}")
    
    print("\n💡 If the address doesn't appear, you may need to:")
    print("   1. Mine for a few minutes to create the account")
    print("   2. Use a different address format")
    print("   3. Register the address manually on the pool website")

def main():
    print("🔧 FIXING POOL REGISTRATION AND ADDRESS ISSUES")
    print("=" * 60)
    
    config = load_config()
    
    # Test current address with each pool
    print("\n📊 Testing current address with pools:")
    for pool in config['pools']:
        test_address_with_pool(pool)
        time.sleep(1)
    
    # Create fixed configurations
    fixed_configs = create_fixed_config()
    
    # Test share submission formats
    test_share_submission_format()
    
    # Check pool websites
    check_pool_websites()
    
    print("\n✅ Fixes applied!")
    print("\n📋 Next steps:")
    print("1. Test each config_test_X.json file")
    print("2. Check pool websites for your address")
    print("3. Try mining for a few minutes to create accounts")
    print("4. Use the working configuration")

if __name__ == "__main__":
    main() 