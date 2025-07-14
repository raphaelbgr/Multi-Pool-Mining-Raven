#!/usr/bin/env python3
"""
Test script to verify nonce formatting fixes for all pools
"""

import json
from adapters.factory import AdapterFactory

def test_nonce_formatting():
    """Test that all pools now use correct little-endian nonce formatting"""
    
    with open("config.json") as f:
        config = json.load(f)
    
    # Test with a sample nonce (same as in logs)
    test_nonce = 4212448  # This was rejected by 2Miners as "malformed nonce"
    
    print(f"Testing nonce formatting for nonce: {test_nonce}")
    print(f"Expected little-endian: 60 4d 40 00 = 604d4000")
    print(f"Wrong big-endian:      00 40 4d 60 = 00404d60")
    print()
    
    for pool_config in config['pools']:
        pool_name = pool_config['name']
        
        try:
            adapter = AdapterFactory.create_adapter(pool_config)
            formatted_nonce = adapter._format_nonce(test_nonce)
            
            # Check if it's little-endian (should start with 60)
            if formatted_nonce.startswith('604d40'):
                status = "[OK] Little-endian"
            elif formatted_nonce.startswith('00404d'):
                status = "[ERROR] Still big-endian"
            else:
                status = "[UNKNOWN] Different format"
            
            print(f"{pool_name:12}: {formatted_nonce} {status}")
            
        except Exception as e:
            print(f"{pool_name:12}: ERROR - {e}")
    
    print()
    print("Expected result: All pools should show little-endian format (604d4000)")
    print("This should fix the 'Malformed nonce' errors from 2Miners and other pools!")

if __name__ == "__main__":
    test_nonce_formatting() 