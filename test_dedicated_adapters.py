#!/usr/bin/env python3
"""
Test script to verify all dedicated pool adapters
"""

from adapters.factory import AdapterFactory
import json

def test_dedicated_adapters():
    # Load configuration
    try:
        with open('config.json') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ config.json not found")
        return False
    
    print("🔧 Testing Dedicated Pool Adapters")
    print("=" * 40)
    
    success_count = 0
    total_pools = len(config['pools'])
    
    for pool_config in config['pools']:
        pool_name = pool_config['name']
        
        try:
            # Test adapter creation
            adapter = AdapterFactory.create_adapter(pool_config)
            print(f"✅ {pool_name}: {adapter.__class__.__name__}")
            success_count += 1
            
            # Test adapter methods exist
            required_methods = ['_perform_handshake', '_parse_job', '_format_submission']
            for method in required_methods:
                if hasattr(adapter, method):
                    print(f"   ✓ {method}")
                else:
                    print(f"   ❌ Missing {method}")
            
        except Exception as e:
            print(f"❌ {pool_name}: Error - {e}")
    
    print("=" * 40)
    print(f"📊 Results: {success_count}/{total_pools} adapters working")
    
    if success_count == total_pools:
        print("🎉 All dedicated adapters are working!")
        return True
    else:
        print("⚠️ Some adapters have issues")
        return False

if __name__ == "__main__":
    test_dedicated_adapters() 