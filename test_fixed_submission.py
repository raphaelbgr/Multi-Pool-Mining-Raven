#!/usr/bin/env python3
"""
Teste da submissão corrigida
"""

import socket
import json
import time

def test_2miners_fixed():
    """Testa 2Miners com formato corrigido"""
    
    print("Testando 2Miners com formato corrigido...")
    print("=" * 50)
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        print("Conectando a rvn.2miners.com:6060...")
        sock.connect(("rvn.2miners.com", 6060))
        print("✅ Conectado!")
        
        # Subscribe
        subscribe_msg = {
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }
        
        print("Enviando subscribe...")
        sock.sendall(json.dumps(subscribe_msg).encode() + b"\n")
        
        response = sock.recv(4096).decode()
        print(f"Resposta subscribe: {response}")
        
        # Parse extra nonce
        subscribe_data = json.loads(response)
        extra_nonce = subscribe_data.get('result', [None, ""])[1]
        print(f"Extra nonce: {extra_nonce}")
        
        # Authorize - formato corrigido
        auth_msg = {
            "id": 2,
            "method": "mining.authorize",
            "params": ["RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU"]
        }
        
        print("Enviando authorize...")
        sock.sendall(json.dumps(auth_msg).encode() + b"\n")
        
        response = sock.recv(4096).decode()
        print(f"Resposta authorize: {response}")
        
        # Aguardar jobs
        print("Aguardando jobs...")
        time.sleep(2)
        
        # Ler jobs disponíveis
        while sock.in_waiting > 0:
            job_data = sock.recv(4096).decode()
            print(f"Job data: {job_data}")
        
        # Testar submit com dados reais
        extranonce2 = "00000001"
        ntime = "00000000"
        nonce = "0014a000"  # 1350272 em hex
        
        submit_msg = {
            "id": 3,
            "method": "mining.submit",
            "params": [
                "RGpVV4jjVhBvKYhT4kouUcndfr3VpwgxVU",
                "45fee",
                extra_nonce + extranonce2,
                ntime,
                nonce
            ]
        }
        
        print("Enviando submit...")
        print(f"Params: {submit_msg['params']}")
        sock.sendall(json.dumps(submit_msg).encode() + b"\n")
        
        response = sock.recv(4096).decode()
        print(f"Resposta submit: {response}")
        
        sock.close()
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_2miners_fixed() 