#!/usr/bin/env python3
"""
Teste manual de submissão de shares
"""

import socket
import json
import time
from adapters import AdapterFactory

def test_2miners_submission():
    """Testa submissão específica para 2Miners"""
    
    print("Testando submissão para 2Miners...")
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
    
    # Job de teste (do log)
    job = {
        "job_id": "45fee",
        "header_hash": "9247ff513d22d261",
        "target": "00000000ffff0000",
        "ntime": "00000000"
    }
    
    nonce = 1350272
    
    try:
        # Criar adapter
        adapter = AdapterFactory.create_adapter(config)
        
        print(f"Conectando a {config['host']}:{config['port']}...")
        adapter.connect()
        print("✅ Conectado!")
        
        # Fazer handshake
        print("Fazendo handshake...")
        adapter._perform_handshake()
        print("✅ Handshake completo!")
        
        # Testar submissão
        print(f"Submetendo share...")
        print(f"  Job ID: {job['job_id']}")
        print(f"  Nonce: {nonce}")
        print(f"  Header: {job['header_hash']}")
        
        result = adapter.submit_share(nonce, job)
        print(f"Resultado: {result}")
        
        adapter.disconnect()
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()

def test_raw_connection():
    """Testa conexão raw com 2Miners"""
    
    print("\nTestando conexão raw...")
    print("=" * 50)
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        print("Conectando a rvn.2miners.com:6060...")
        sock.connect(("rvn.2miners.com", 6060))
        print("✅ Conectado!")
        
        # Enviar subscribe
        subscribe_msg = {
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }
        
        print("Enviando subscribe...")
        sock.sendall(json.dumps(subscribe_msg).encode() + b"\n")
        
        response = sock.recv(4096).decode()
        print(f"Resposta subscribe: {response}")
        
        # Enviar authorize
        auth_msg = {
            "id": 2,
            "method": "mining.authorize",
            "params": ["RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"]
        }
        
        print("Enviando authorize...")
        sock.sendall(json.dumps(auth_msg).encode() + b"\n")
        
        response = sock.recv(4096).decode()
        print(f"Resposta authorize: {response}")
        
        # Testar submit
        submit_msg = {
            "id": 3,
            "method": "mining.submit",
            "params": [
                "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU",
                "45fee",
                "00000001",
                "00000000",
                "0014a000"
            ]
        }
        
        print("Enviando submit...")
        sock.sendall(json.dumps(submit_msg).encode() + b"\n")
        
        response = sock.recv(4096).decode()
        print(f"Resposta submit: {response}")
        
        sock.close()
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()

def test_other_pools():
    """Testa outras pools para comparação"""
    
    print("\nTestando outras pools...")
    print("=" * 50)
    
    pools = [
        {
            "name": "Ravenminer",
            "host": "stratum.ravenminer.com",
            "port": 3838,
            "user": "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU.Yoda"
        },
        {
            "name": "Nanopool",
            "host": "rvn-us-east1.nanopool.org",
            "port": 10400,
            "user": "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU.Luke"
        }
    ]
    
    for pool in pools:
        print(f"\nTestando {pool['name']}...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            sock.connect((pool['host'], pool['port']))
            print(f"✅ Conectado a {pool['name']}")
            
            # Subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            
            sock.sendall(json.dumps(subscribe_msg).encode() + b"\n")
            response = sock.recv(4096).decode()
            print(f"Subscribe OK: {response[:100]}...")
            
            sock.close()
            
        except Exception as e:
            print(f"❌ Erro com {pool['name']}: {str(e)}")

if __name__ == "__main__":
    print("🔧 Teste de Submissão de Shares")
    print("=" * 50)
    
    # Testar 2Miners especificamente
    test_2miners_submission()
    
    # Testar conexão raw
    test_raw_connection()
    
    # Testar outras pools
    test_other_pools()
    
    print("\n✅ Testes concluídos!") 