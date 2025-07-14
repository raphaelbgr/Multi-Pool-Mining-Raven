import json
from .base import PoolAdapter

class Stratum1Adapter2Miners(PoolAdapter):
    """Adapter específico para 2Miners com formato corrigido"""
    
    def _perform_handshake(self):
        # Subscribe
        self.sock.sendall(json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }).encode() + b"\n")
        
        # 2Miners pode enviar múltiplas respostas, processar linha por linha
        subscribe_response = self.sock.recv(4096).decode()
        
        # Encontrar resposta do subscribe
        for line in subscribe_response.split('\n'):
            if line.strip():
                try:
                    response = json.loads(line)
                    if response.get('id') == 1:  # Resposta do subscribe
                        self.extra_nonce = response.get('result', [None, ""])[1]
                        break
                except json.JSONDecodeError:
                    continue
        
        # Authorize - formato específico para 2Miners
        # 2Miners requer: [user, password] mesmo que password seja "x"
        auth_params = [self.config['user'], self.config['password']]
            
        self.sock.sendall(json.dumps({
            "id": 2,
            "method": "mining.authorize",
            "params": auth_params
        }).encode() + b"\n")
        
        # 2Miners pode enviar múltiplas respostas, ler até encontrar a autorização
        auth_response = self.sock.recv(4096).decode()
        
        # Dividir respostas múltiplas por linha
        for line in auth_response.split('\n'):
            if line.strip():
                try:
                    auth_data = json.loads(line)
                    if auth_data.get('id') == 2:  # Resposta da autorização
                        if auth_data.get('result') == True:
                            print("[OK] 2Miners authorization successful")
                        else:
                            print(f"[WARN] 2Miners auth result: {auth_data}")
                        break
                except json.JSONDecodeError:
                    continue

    def _parse_job(self, params):
        return {
            'job_id': params[0],
            'header_hash': params[1],
            'target': params[3],
            'ntime': self._format_ntime(params[6] if len(params) > 6 else None)
        }

    def _format_ntime(self, ntime):
        if ntime is None:
            return "00000000"
        if isinstance(ntime, int):
            return format(ntime, '08x')
        elif isinstance(ntime, str) and len(ntime) == 7:
            return '0' + ntime
        return str(ntime)

    def _format_nonce(self, nonce):
        """Format nonce in big-endian format for KawPOW on 2Miners"""
        # 2Miners expects big-endian nonce for KawPOW
        return f"{nonce:08x}"

    def _format_submission(self, nonce, job):
        extranonce2 = "00000001"
        if 'extranonce_size' in self.config['extra']:
            extranonce2_size = self.config['extra']['extranonce_size']
            # extranonce2 should be extranonce2_size BYTES, so extranonce2_size * 2 hex chars
            extranonce2 = extranonce2[:extranonce2_size * 2].ljust(extranonce2_size * 2, '0')
            
        return {
            "id": self.job_counter + 1,
            "method": self.config['extra'].get('submit_method', 'mining.submit'),
            "params": [
                self.config['user'],
                job['job_id'],
                extranonce2,
                job['ntime'],
                self._format_nonce(nonce)
            ]
        } 