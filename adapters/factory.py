from .stratum1 import Stratum1Adapter
from .stratum2 import Stratum2Adapter

class AdapterFactory:
    @staticmethod
    def create_adapter(pool_config):
        adapter_type = pool_config.get('adapter', 'stratum1').lower()
        
        if adapter_type == 'stratum1':
            return Stratum1Adapter(pool_config)
        elif adapter_type == 'stratum2':
            return Stratum2Adapter(pool_config)
        
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    @staticmethod
    def supported_adapters():
        return ['stratum1', 'stratum2']