import socket
import json
import time
from abc import ABC, abstractmethod

class PoolAdapter(ABC):
    def __init__(self, pool_config):
        self.config = pool_config
        self.socket_timeout = 10
        self.connected = False
        self.extra_nonce = ""
        self.sock = None
        
    def connect(self):
        try:
            self.sock = socket.create_connection(
                (self.config['host'], self.config['port']),
                timeout=self.socket_timeout
            )
            self.sock.settimeout(self.socket_timeout)
            self._perform_handshake()
            self.connected = True
        except Exception as e:
            print(f"❌ Connection failed to {self.config['name']}: {str(e)}")
            self.connected = False
            
    @abstractmethod
    def _perform_handshake(self):
        pass
        
    @abstractmethod
    def get_job(self):
        pass
        
    @abstractmethod
    def submit_share(self, nonce, job):
        pass
        
    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False

class Stratum1Adapter(PoolAdapter):
    def _perform_handshake(self):
        # Subscribe
        self.sock.sendall(json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": []
        }).encode() + b"\n")
        
        response = self._receive_response()
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
        self._receive_response()
        
    def get_job(self):
        while True:
            response = self._receive_response()
            if response.get('method') == 'mining.notify':
                params = response['params']
                return {
                    'job_id': params[0],
                    'header_hash': params[1],
                    'target': params[3],
                    'ntime': self._format_ntime(params[6] if len(params) > 6 else None)
                }
                
    def _format_ntime(self, ntime):
        if isinstance(ntime, int):
            return format(ntime, '08x')
        elif len(ntime) == 7:  # Some pools send 7-char ntime
            return '0' + ntime
        return ntime
        
    def submit_share(self, nonce, job):
        extranonce2 = "00000001"
        if 'extranonce_size' in self.config['extra']:
            extranonce2 = extranonce2[:self.config['extra']['extranonce_size']]
            
        payload = {
            "id": 4,
            "method": self.config['extra'].get('submit_method', 'mining.submit'),
            "params": [
                self.config['user'],
                job['job_id'],
                self.extra_nonce + extranonce2,
                job['ntime'],
                format(nonce, '08x')
            ]
        }
        self.sock.sendall((json.dumps(payload) + "\n").encode())
        return self._receive_response()
        
    def _receive_response(self):
        data = self.sock.recv(4096).decode()
        try:
            return json.loads(data.splitlines()[0])
        except:
            return {"error": "Invalid JSON response"}

class Stratum2Adapter(Stratum1Adapter):
    def _perform_handshake(self):
        # Different initial handshake for some pools
        self.sock.sendall(json.dumps({
            "id": 1,
            "method": "mining.subscribe",
            "params": [f"{self.config['name']}-Client/1.0.0"]
        }).encode() + b"\n")
        
        response = self._receive_response()
        self.extra_nonce = response.get('result', [None, ""])[1]
        
        # Authorize with worker name if required
        auth_params = [self.config['user']]
        if self.config['extra'].get('require_worker', True):
            auth_params.append(self.config['password'])
            
        self.sock.sendall(json.dumps({
            "id": 2,
            "method": "mining.authorize",
            "params": auth_params
        }).encode() + b"\n")
        self._receive_response()
        
    def get_job(self):
        while True:
            response = self._receive_response()
            if response.get('method') == 'mining.notify':
                params = response['params']
                return {
                    'job_id': params[0],
                    'header_hash': params[1],
                    'target': params[3],
                    'ntime': self._format_ntime(params[5] if len(params) > 5 else None)
                }

class AdapterFactory:
    @staticmethod
    def create_adapter(pool_config):
        adapter_type = pool_config.get('adapter', 'stratum1')
        if adapter_type == 'stratum1':
            return Stratum1Adapter(pool_config)
        elif adapter_type == 'stratum2':
            return Stratum2Adapter(pool_config)
        raise ValueError(f"Unknown adapter type: {adapter_type}")