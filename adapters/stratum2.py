import json
from .base import PoolAdapter

class Stratum2Adapter(PoolAdapter):
    def _perform_handshake(self):
        self.sock.sendall(json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": [f"{self.config['name']}-Client/1.0.0"]
        }).encode() + b"\n")
        
        # Handle subscribe response
        response = self.sock.recv(4096).decode().strip()
        self.extra_nonce = ""
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if parsed.get('id') == 1 and 'result' in parsed:
                    result = parsed['result']
                    if isinstance(result, list) and len(result) >= 2:
                        self.extra_nonce = result[1] if result[1] else ""
                    break
            except json.JSONDecodeError:
                continue
        
        auth_params = [self.config['user']]
        if self.config['extra'].get('require_worker', True):
            auth_params.append(self.config['password'])
            
        self.sock.sendall(json.dumps({
            "id": 2,
            "method": "mining.authorize",
            "params": auth_params
        }).encode() + b"\n")
        
        # Handle auth response
        auth_response = self.sock.recv(4096).decode().strip()
        print(f"Auth response: {auth_response}")  # Debug
        
        # Parse auth response to check for success
        auth_success = False
        for line in auth_response.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if parsed.get('id') == 2:
                    auth_success = parsed.get('result', False)
                    if parsed.get('error'):
                        print(f"[ERROR] Auth error: {parsed['error']}")
                    break
            except json.JSONDecodeError:
                continue
                
        if not auth_success:
            print(f"[WARN] Authorization may have failed for {self.config['name']}")
        else:
            print(f"[OK] Authorization successful for {self.config['name']}")

    def _parse_job(self, params):
        return {
            'job_id': params[0],
            'header_hash': params[1],
            'target': params[3],
            'ntime': self._format_ntime(params[5] if len(params) > 5 else None)
        }

    def _format_ntime(self, ntime):
        if isinstance(ntime, int):
            return format(ntime, '08x')
        return ntime

    def _format_nonce(self, nonce):
        """Format nonce in little-endian format for Ravencoin"""
        # Convert to little-endian bytes
        byte0 = (nonce >> 0) & 0xFF
        byte1 = (nonce >> 8) & 0xFF
        byte2 = (nonce >> 16) & 0xFF
        byte3 = (nonce >> 24) & 0xFF
        
        # Format as little-endian hex string
        return f"{byte0:02x}{byte1:02x}{byte2:02x}{byte3:02x}"

    def _format_submission(self, nonce, job):
        return {
            "id": self.job_counter + 1,
            "method": "mining.submit",
            "params": [
                self.config['user'],
                job['job_id'],
                self.config['extra'].get('fixed_extranonce', '00000000'),
                job['ntime'],
                self._format_nonce(nonce)
            ]
        }