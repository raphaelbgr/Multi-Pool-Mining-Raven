import json
from .base import PoolAdapter

class Stratum1Adapter(PoolAdapter):
    def _perform_handshake(self):
        # Subscribe
        self.sock.sendall(json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": []
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
        
        # Authorize - all pools get user and password
        auth_params = [self.config['user']]
        if self.config['extra'].get('require_worker', False) or self.config['password'] != 'x':
            auth_params.append(self.config['password'])
        else:
            auth_params.append(self.config['password'])  # Always include password
            
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
            'ntime': self._format_ntime(params[6] if len(params) > 6 else None)
        }

    def _format_ntime(self, ntime):
        if isinstance(ntime, int):
            return format(ntime, '08x')
        elif len(ntime) == 7:
            return '0' + ntime
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
        # Calculate extranonce2 based on pool requirements
        if 'extranonce_size' in self.config['extra']:
            extranonce2_size = self.config['extra']['extranonce_size']
            extranonce2 = "00000001"[:extranonce2_size * 2]  # Each byte = 2 hex chars
            extranonce2 = extranonce2.ljust(extranonce2_size * 2, '0')
        else:
            extranonce2 = "00000001"  # Default 4-byte extranonce2
            
        return {
            "id": self.job_counter + 1,
            "method": self.config['extra'].get('submit_method', 'mining.submit'),
            "params": [
                self.config['user'],
                job['job_id'],
                self.extra_nonce + extranonce2,
                job['ntime'],
                self._format_nonce(nonce)
            ]
        }