import json
import socket
from .base import PoolAdapter

class PoolRavenminerAdapter(PoolAdapter):
    """Fixed adapter for Ravenminer pool with X16R algorithm support"""
    
    def _perform_handshake(self):
        # Ravenminer-specific subscribe
        self.sock.sendall(json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }).encode() + b"\n")
        
        # Handle subscribe response
        response = self.sock.recv(4096).decode()
        for line in response.split('\n'):
            if line.strip():
                try:
                    parsed = json.loads(line)
                    if parsed.get('id') == 1:
                        result = parsed.get('result', [])
                        if len(result) >= 2:
                            self.extra_nonce = result[1] or ""
                        break
                except json.JSONDecodeError:
                    continue
        
        # Ravenminer authorization
        self.sock.sendall(json.dumps({
            "id": 2,
            "method": "mining.authorize",
            "params": [self.config['user'], self.config['password']]
        }).encode() + b"\n")
        
        # Handle auth response
        auth_response = self.sock.recv(4096).decode()
        print(f"[OK] Ravenminer authorization: {auth_response.strip()}")

    def _parse_job(self, params):
        return {
            'job_id': params[0],
            'header_hash': params[1],
            'target': params[3],
            'ntime': params[6] if len(params) > 6 else "00000000"
        }

    def _format_nonce(self, nonce):
        """Ravenminer expects X16R little-endian nonce"""
        # X16R algorithm requires little-endian nonce for proper hashing
        return nonce.to_bytes(4, 'little').hex()

    def _format_submission(self, nonce, job):
        # Use actual T-Rex extranonce2 instead of hardcoded value
        trex_extranonce2 = self._get_extranonce2_from_trex(job)
        
        # Ravenminer expects 2-byte extranonce2 (4 hex chars), use last 4 chars
        if len(trex_extranonce2) >= 4:
            extranonce2_part = trex_extranonce2[-4:]  # Use last 4 chars
        else:
            # If T-Rex extranonce2 is shorter, pad with zeros
            extranonce2_part = trex_extranonce2.ljust(4, '0')
        
        # Use pool's extra_nonce + extranonce2 (X16R protocol)
        full_extranonce = self.extra_nonce + extranonce2_part
        
        return {
            "id": self.job_counter + 1,
            "method": "mining.submit",
            "params": [
                self.config['user'],
                job['job_id'],
                full_extranonce,
                job['ntime'],
                self._format_nonce(nonce)
            ]
        }