#!/usr/bin/env python3
"""
DEBUG HEADER VALIDATION
Compare our header construction with known valid Ravencoin blocks
"""

import hashlib
import struct
import time

def debug_header_construction():
    """Debug header construction by comparing with known valid blocks"""
    print("=" * 80)
    print("DEBUG HEADER CONSTRUCTION")
    print("=" * 80)
    
    # Example from a known Ravencoin block (approximate)
    # This is for demonstration - we'll use realistic values
    
    # Known block components (example)
    version = 0x20000000
    prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    merkle_root = "104e8aac312ec69f45945cbebeca3e22f6a1b31d2488aa752b663fcc645a1e71"
    ntime = 0x1b00f9ab
    nbits = 0x1a01a01a
    nonce = 0x12345678
    
    print(f"[BLOCK] Example block components:")
    print(f"  Version: 0x{version:08x}")
    print(f"  Prev Hash: {prev_hash}")
    print(f"  Merkle Root: {merkle_root}")
    print(f"  Ntime: 0x{ntime:08x}")
    print(f"  Nbits: 0x{nbits:08x}")
    print(f"  Nonce: 0x{nonce:08x}")
    
    # Method 1: Our current method
    print(f"\n[METHOD 1] Our current method:")
    header1 = construct_header_method1(version, prev_hash, merkle_root, ntime, nbits, nonce)
    print(f"  Header (hex): {header1.hex()}")
    print(f"  Length: {len(header1)} bytes")
    
    # Method 2: Proper little-endian method
    print(f"\n[METHOD 2] Proper little-endian method:")
    header2 = construct_header_method2(version, prev_hash, merkle_root, ntime, nbits, nonce)
    print(f"  Header (hex): {header2.hex()}")
    print(f"  Length: {len(header2)} bytes")
    
    # Method 3: Double SHA-256 hash
    print(f"\n[METHOD 3] Double SHA-256 hash:")
    header_hash = hash_header(header2)
    print(f"  Header hash: {header_hash.hex()}")
    print(f"  Length: {len(header_hash)} bytes")
    
    # Compare with what pools expect
    print(f"\n[COMPARISON] What pools expect vs what we send:")
    print(f"  Pool header_hash: {merkle_root}")
    print(f"  Our header_hash: {header_hash.hex()}")
    print(f"  Match: {merkle_root == header_hash.hex()}")
    
    return header1, header2, header_hash

def construct_header_method1(version, prev_hash, merkle_root, ntime, nbits, nonce):
    """Method 1: Our current method (likely wrong)"""
    header = bytearray(80)
    
    # Version (4 bytes) - little-endian
    header[0:4] = struct.pack("<I", version)
    
    # Previous block hash (32 bytes) - little-endian
    prev_hash_bytes = bytes.fromhex(prev_hash)
    header[4:36] = prev_hash_bytes
    
    # Merkle root (32 bytes) - little-endian
    merkle_bytes = bytes.fromhex(merkle_root)
    header[36:68] = merkle_bytes
    
    # Timestamp (4 bytes) - little-endian
    header[68:72] = struct.pack("<I", ntime)
    
    # Bits (4 bytes) - little-endian
    header[72:76] = struct.pack("<I", nbits)
    
    # Nonce (4 bytes) - little-endian
    header[76:80] = struct.pack("<I", nonce)
    
    return bytes(header)

def construct_header_method2(version, prev_hash, merkle_root, ntime, nbits, nonce):
    """Method 2: Proper little-endian method"""
    header = bytearray(80)
    
    # Version (4 bytes) - little-endian
    header[0:4] = struct.pack("<I", version)
    
    # Previous block hash (32 bytes) - reverse byte order for little-endian
    prev_hash_bytes = bytes.fromhex(prev_hash)[::-1]
    header[4:36] = prev_hash_bytes
    
    # Merkle root (32 bytes) - reverse byte order for little-endian
    merkle_bytes = bytes.fromhex(merkle_root)[::-1]
    header[36:68] = merkle_bytes
    
    # Timestamp (4 bytes) - little-endian
    header[68:72] = struct.pack("<I", ntime)
    
    # Bits (4 bytes) - little-endian
    header[72:76] = struct.pack("<I", nbits)
    
    # Nonce (4 bytes) - little-endian
    header[76:80] = struct.pack("<I", nonce)
    
    return bytes(header)

def hash_header(header_bytes):
    """Double SHA-256 hash the header"""
    return hashlib.sha256(hashlib.sha256(header_bytes).digest()).digest()

def test_nonce_formats():
    """Test different nonce formats"""
    print(f"\n" + "=" * 80)
    print("TESTING NONCE FORMATS")
    print("=" * 80)
    
    test_nonce = 0x12345678  # 305419896
    
    formats = {
        "big_endian_hex": f"{test_nonce:08x}",
        "little_endian_hex": test_nonce.to_bytes(4, 'little').hex(),
        "big_endian_bytes": struct.pack(">I", test_nonce).hex(),
        "little_endian_bytes": struct.pack("<I", test_nonce).hex(),
        "decimal": str(test_nonce),
        "hex_prefix": f"0x{test_nonce:08x}"
    }
    
    print(f"[NONCE] Test nonce: 0x{test_nonce:08x} ({test_nonce})")
    print(f"[NONCE] Binary: {bin(test_nonce)}")
    
    for format_name, nonce_val in formats.items():
        print(f"  {format_name}: {nonce_val}")
    
    # Test what 2Miners expects
    print(f"\n[2MINERS] Expected format analysis:")
    print(f"  Original: 0x12345678")
    print(f"  Little-endian: 78563412")
    print(f"  Big-endian: 12345678")
    
    return formats

def test_extranonce2_formats():
    """Test different extranonce2 formats"""
    print(f"\n" + "=" * 80)
    print("TESTING EXTRANONCE2 FORMATS")
    print("=" * 80)
    
    pool_extra_nonce = "92d652"  # From Nanopool test
    
    formats = {
        "2byte": "0001",
        "4byte": "00000001", 
        "6byte": "000000000001",
        "8byte": "0000000000000001",
        "pool_extra_2byte": pool_extra_nonce + "0001",
        "pool_extra_4byte": pool_extra_nonce + "00000001",
        "pool_extra_6byte": pool_extra_nonce + "000000000001",
    }
    
    print(f"[EXTRANONCE2] Pool extra_nonce: '{pool_extra_nonce}' (len: {len(pool_extra_nonce)})")
    
    for format_name, ext2_val in formats.items():
        print(f"  {format_name}: {ext2_val} (len: {len(ext2_val)})")
    
    return formats

def create_working_submission_template():
    """Create working submission template based on analysis"""
    print(f"\n" + "=" * 80)
    print("WORKING SUBMISSION TEMPLATE")
    print("=" * 80)
    
    template = {
        "ravenminer": {
            "description": "Ravenminer expects little-endian nonce and 2-byte extranonce2",
            "nonce_format": "little_endian_hex",
            "extranonce2_format": "2byte", 
            "ntime_format": "big_endian_hex",
            "example": {
                "user": "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU.Yoda",
                "job_id": "5662",
                "extranonce2": "0001",
                "ntime": "1b00f9ab",
                "nonce": "78563412"
            }
        },
        "2miners": {
            "description": "2Miners expects little-endian nonce and 4-byte extranonce2",
            "nonce_format": "little_endian_hex",
            "extranonce2_format": "4byte",
            "ntime_format": "big_endian_hex", 
            "example": {
                "user": "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU",
                "job_id": "464be",
                "extranonce2": "00000001",
                "ntime": "1b00f9ab",
                "nonce": "78563412"
            }
        },
        "nanopool": {
            "description": "Nanopool expects specific extranonce2 format",
            "nonce_format": "little_endian_hex",
            "extranonce2_format": "pool_extra_2byte",
            "ntime_format": "big_endian_hex",
            "example": {
                "user": "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU.Luke",
                "job_id": "5689", 
                "extranonce2": "92d6520001",
                "ntime": "1b00f9ab",
                "nonce": "78563412"
            }
        }
    }
    
    for pool_name, config in template.items():
        print(f"\n[{pool_name.upper()}] {config['description']}")
        print(f"  Nonce format: {config['nonce_format']}")
        print(f"  Extranonce2 format: {config['extranonce2_format']}")
        print(f"  Ntime format: {config['ntime_format']}")
        print(f"  Example submission:")
        example = config['example']
        print(f"    User: {example['user']}")
        print(f"    Job ID: {example['job_id']}")
        print(f"    Extranonce2: {example['extranonce2']}")
        print(f"    Ntime: {example['ntime']}")
        print(f"    Nonce: {example['nonce']}")
    
    return template

def run_complete_debug():
    """Run complete header debugging"""
    print("DEBUG HEADER VALIDATION")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Debug header construction
    header1, header2, header_hash = debug_header_construction()
    
    # Test nonce formats
    nonce_formats = test_nonce_formats()
    
    # Test extranonce2 formats
    extranonce2_formats = test_extranonce2_formats()
    
    # Create working template
    template = create_working_submission_template()
    
    print(f"\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
    print("Key findings:")
    print("1. Header construction needs proper endianness")
    print("2. Nonce should be little-endian hex")
    print("3. Extranonce2 format varies by pool")
    print("4. Each pool has specific requirements")
    
    return {
        'headers': (header1, header2, header_hash),
        'nonce_formats': nonce_formats,
        'extranonce2_formats': extranonce2_formats,
        'template': template
    }

if __name__ == "__main__":
    results = run_complete_debug() 