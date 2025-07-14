#!/usr/bin/env python3
"""
Teste final da correção para 2Miners
"""

from adapters import AdapterFactory
import json

def test_2miners_corrected():
    """Testa 2Miners com correção aplicada"""
    
    print("🔧 Teste Final - 2Miners Corrigido")
    print("=" * 50)
    
    # Configuração da 2Miners
    config = {
        "name": "2Miners",
        "host": "rvn.2miners.com",
        "port": 6060,
        "user": "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU",
        "password": "x",
        "adapter": "stratum1",
        "extra": {
            "extranonce_size": 4,
            "submit_method": "mining.submit",
            "require_worker": False
        }
    }
    
    # Job de teste
    job = {
        "job_id": "45ff0",
        "header_hash": "921373a8d9d91545",
        "target": "00000000ffff0000",
        "ntime": "00000000"
    }
    
    nonce = 1350272
    
    try:
        print(f"Criando adapter para {config['name']}...")
        adapter = AdapterFactory.create_adapter(config)
        print(f"✅ Adapter criado: {type(adapter).__name__}")
        
        print(f"Conectando a {config['host']}:{config['port']}...")
        adapter.connect()
        print("✅ Conectado!")
        
        print("Fazendo handshake...")
        adapter._perform_handshake()
        print("✅ Handshake completo!")
        
        print(f"Submetendo share...")
        print(f"  Job ID: {job['job_id']}")
        print(f"  Nonce: {nonce} (0x{nonce:08x})")
        print(f"  Header: {job['header_hash']}")
        
        result = adapter.submit_share(nonce, job)
        print(f"✅ Resultado: {result}")
        
        adapter.close()
        print("✅ Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_2miners_corrected() 