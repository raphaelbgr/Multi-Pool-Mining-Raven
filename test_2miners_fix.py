#!/usr/bin/env python3
"""
Test script to verify 2Miners nonce formatting and extranonce fixes
"""

import json
from adapters.factory import AdapterFactory

def test_2miners_formatting():
    """Test that 2Miners now formats submissions correctly"""
    
    with open("config.json") as f:
        config = json.load(f)
    
    # Test with the same nonce that was failing
    test_nonce = 4212448
    
    # Create sample job data
    sample_job = {
        'job_id': 'test_job_123',
        'header_hash': 'abcdef0123456789' * 8,
        'target': '00000004fb000000' + '0' * 48,
        'ntime': '12345678',
        'trex_extranonce2': '0xae00000012345678'  # Sample T-Rex extranonce2
    }
    
    print("Testing 2Miners submission formatting...")
    print(f"Test nonce: {test_nonce}")
    print(f"Expected nonce format: e0464000 (little-endian)")
    print()
    
    # Test 2Miners adapter
    miners_pool = next(pool for pool in config['pools'] if pool['name'] == '2Miners')
    miners_adapter = AdapterFactory.create_adapter(miners_pool)
    
    # Test nonce formatting
    formatted_nonce = miners_adapter._format_nonce(test_nonce)
    print(f"2Miners nonce format: {formatted_nonce}")
    
    # Test full submission format
    submission = miners_adapter._format_submission(test_nonce, sample_job)
    print(f"2Miners submission:")
    print(f"  Method: {submission['method']}")
    print(f"  User: {submission['params'][0]}")
    print(f"  Job ID: {submission['params'][1]}")
    print(f"  Extranonce: {submission['params'][2]}")
    print(f"  Ntime: {submission['params'][3]}")
    print(f"  Nonce: {submission['params'][4]}")
    
    # Compare with Ravenminer (working) adapter
    print("\nComparing with Ravenminer (working):")
    ravenminer_pool = next(pool for pool in config['pools'] if pool['name'] == 'Ravenminer')
    ravenminer_adapter = AdapterFactory.create_adapter(ravenminer_pool)
    
    ravenminer_submission = ravenminer_adapter._format_submission(test_nonce, sample_job)
    print(f"Ravenminer submission:")
    print(f"  Method: {ravenminer_submission['method']}")
    print(f"  User: {ravenminer_submission['params'][0]}")
    print(f"  Job ID: {ravenminer_submission['params'][1]}")
    print(f"  Extranonce: {ravenminer_submission['params'][2]}")
    print(f"  Ntime: {ravenminer_submission['params'][3]}")
    print(f"  Nonce: {ravenminer_submission['params'][4]}")
    
    # Check if both use the same nonce format
    if formatted_nonce == ravenminer_adapter._format_nonce(test_nonce):
        print("\n✅ Both adapters use the same nonce format!")
    else:
        print("\n❌ Nonce formats differ!")
    
    # Check if both use extra_nonce + extranonce2
    if len(submission['params'][2]) > 8 and len(ravenminer_submission['params'][2]) > 4:
        print("✅ Both adapters use full extranonce (extra_nonce + extranonce2)")
    else:
        print("❌ Extranonce formats may differ!")

if __name__ == "__main__":
    test_2miners_formatting() 