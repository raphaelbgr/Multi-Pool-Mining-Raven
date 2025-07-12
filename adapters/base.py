import socket
import json
from abc import ABC, abstractmethod

class PoolAdapter(ABC):
    def __init__(self, pool_config):
        self.config = pool_config
        self.socket_timeout = 10
        self.connected = False
        self.extra_nonce = ""
        self.sock = None
        self.job_counter = 0

    @abstractmethod
    def _perform_handshake(self):
        pass
        
    @abstractmethod
    def _parse_job(self, params):
        pass
        
    @abstractmethod
    def _format_submission(self, nonce, job):
        pass

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

    def get_job(self):
        if not self.connected:
            self.connect()
            if not self.connected:
                return None

        while True:
            try:
                data = self.sock.recv(4096).decode()
                for line in data.splitlines():
                    try:
                        msg = json.loads(line)
                        if msg.get('method') == 'mining.notify':
                            return self._parse_job(msg['params'])
                    except json.JSONDecodeError:
                        continue
            except socket.timeout:
                return None
            except Exception as e:
                print(f"❌ Error getting job from {self.config['name']}: {str(e)}")
                return None

    def submit_share(self, nonce, job):
        if not self.connected:
            self.connect()
            if not self.connected:
                return False

        try:
            payload = self._format_submission(nonce, job)
            self.sock.sendall((json.dumps(payload) + "\n").encode())
            response = self.sock.recv(4096).decode()
            return json.loads(response)
        except Exception as e:
            print(f"❌ Error submitting to {self.config['name']}: {str(e)}")
            return {"error": str(e)}

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.connected = False