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
        
        response = json.loads(self.sock.recv(4096).decode())
        self.extra_nonce = response.get('result', [None, ""])[1]
        
        # Authorize
        auth_params = [self.config['user']]
        if self.config['extra'].get('require_worker', False):
            auth_params.append(self.config['password'])
            
        self.sock.sendall(json.dumps({
            "id": 2,
            "method": "mining.authorize",
            "params": auth_params
        }).encode() + b"\n")
        json.loads(self.sock.recv(4096).decode())

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

    def _format_submission(self, nonce, job):
        extranonce2 = "00000001"
        if 'extranonce_size' in self.config['extra']:
            extranonce2 = extranonce2[:self.config['extra']['extranonce_size']]
            
        return {
            "id": self.job_counter + 1,
            "method": self.config['extra'].get('submit_method', 'mining.submit'),
            "params": [
                self.config['user'],
                job['job_id'],
                self.extra_nonce + extranonce2,
                job['ntime'],
                format(nonce, '08x')
            ]
        }