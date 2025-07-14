import json
import socket
from .base import PoolAdapter

class PoolNanopoolAdapter(PoolAdapter):
    """Fixed adapter for Nanopool pool with X16R algorithm support"""
    
    def _perform_handshake(self):
        # Nanopool-specific subscribe (immediate without params)
        self.sock.sendall(json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }).encode() + b"\n")
        
        # Handle subscribe response - Nanopool sends target immediately
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
        
        # Read initial data that Nanopool sends
        initial_data = self.sock.recv(4096).decode()
        print(f"Nanopool initial data: {initial_data.strip()}")
        
        # Nanopool authorization
        self.sock.sendall(json.dumps({
            "id": 2,
            "method": "mining.authorize",
            "params": [self.config['user'], self.config['password']]
        }).encode() + b"\n")
        
        # Handle auth response
        auth_response = self.sock.recv(4096).decode()
        print(f"[OK] Nanopool authorization: {auth_response.strip()}")

    def _parse_job(self, params):
        return {
            'job_id': params[0],
            'header_hash': params[1],
            'target': params[3],
            'ntime': params[6] if len(params) > 6 else "00000000"
        }

    def _format_nonce(self, nonce):
        """Nanopool expects X16R little-endian nonce"""
        # X16R algorithm requires little-endian nonce for proper hashing
        return nonce.to_bytes(4, 'little').hex()

    def _format_submission(self, nonce, job):
        # Use actual T-Rex extranonce2 instead of hardcoded value
        trex_extranonce2 = self._get_extranonce2_from_trex(job)
        
        # Nanopool expects exactly pool's extra_nonce + 2 bytes extranonce2
        # From reverse engineering: extra_nonce is 6 chars, total should be around 8-10 chars
        if len(trex_extranonce2) >= 4:
            extranonce2 = trex_extranonce2[-4:]  # Use last 4 chars (2 bytes)
        else:
            extranonce2 = trex_extranonce2.ljust(4, '0')
        
        # Use pool's extra_nonce + our 2-byte extranonce2 (X16R protocol)
        full_extranonce = self.extra_nonce + extranonce2
        
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