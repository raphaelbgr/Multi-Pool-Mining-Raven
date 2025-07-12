import json
from .base import PoolAdapter

class Stratum2Adapter(PoolAdapter):
    def _perform_handshake(self):
        self.sock.sendall(json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": [f"{self.config['name']}-Client/1.0.0"]
        }).encode() + b"\n")
        
        response = json.loads(self.sock.recv(4096).decode())
        self.extra_nonce = response.get('result', [None, ""])[1]
        
        auth_params = [self.config['user']]
        if self.config['extra'].get('require_worker', True):
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
            'ntime': self._format_ntime(params[5] if len(params) > 5 else None)
        }

    def _format_ntime(self, ntime):
        if isinstance(ntime, int):
            return format(ntime, '08x')
        return ntime

    def _format_submission(self, nonce, job):
        return {
            "id": self.job_counter + 1,
            "method": "mining.submit",
            "params": [
                self.config['user'],
                job['job_id'],
                self.config['extra'].get('fixed_extranonce', '00000000'),
                job['ntime'],
                format(nonce, '08x')
            ]
        }