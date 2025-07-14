import json
from .base import PoolAdapter
from .pool_2miners import Pool2MinersAdapter
from .pool_ravenminer import PoolRavenminerAdapter
from .pool_nanopool import PoolNanopoolAdapter
from .pool_woolypooly import PoolWoolyPoolyAdapter
from .pool_herominers import PoolHeroMinersAdapter
from .stratum1 import Stratum1Adapter
from .stratum2 import Stratum2Adapter

class AdapterFactory:
    @staticmethod
    def create_adapter(config):
        """Create appropriate adapter based on pool configuration"""
        pool_name = config['name'].lower()
        
        # Use specific adapters for known pools
        if '2miners' in pool_name:
            return Pool2MinersAdapter(config)
        elif 'ravenminer' in pool_name:
            return PoolRavenminerAdapter(config)
        elif 'nanopool' in pool_name:
            return PoolNanopoolAdapter(config)
        elif 'woolypooly' in pool_name:
            return PoolWoolyPoolyAdapter(config)
        elif 'herominers' in pool_name:
            return PoolHeroMinersAdapter(config)
        else:
            # Fall back to generic adapters
            if config.get('adapter') == 'stratum2':
                return Stratum2Adapter(config)
            else:
                return Stratum1Adapter(config)
    
    @staticmethod
    def supported_pools():
        """Return list of supported pool types"""
        return [
            "2Miners",
            "Ravenminer", 
            "Nanopool",
            "WoolyPooly",
            "HeroMiners",
            "Generic Stratum1",
            "Generic Stratum2"
        ]