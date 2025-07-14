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

    def _get_extranonce2_from_trex(self, job):
        """Extract extranonce2 from T-Rex data, removing 0x prefix if present"""
        trex_extranonce2 = job.get('trex_extranonce2', '')
        if trex_extranonce2.startswith('0x'):
            return trex_extranonce2[2:]
        return trex_extranonce2

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
            print(f"[ERROR] Connection failed to {self.config['name']}: {str(e)}")
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
                print(f"[ERROR] Error getting job from {self.config['name']}: {str(e)}")
                return None

    def submit_share(self, nonce, job):
        if not self.connected:
            self.connect()
            if not self.connected:
                return False

        try:
            payload = self._format_submission(nonce, job)
            self.job_counter += 1
            self.sock.sendall((json.dumps(payload) + "\n").encode())
            
            # Read response with timeout - may need multiple attempts
            target_id = self.job_counter
            response_found = False
            attempts = 0
            
            while not response_found and attempts < 3:
                try:
                    response = self.sock.recv(4096).decode().strip()
                    
                    # Handle empty response
                    if not response:
                        attempts += 1
                        continue
                        
                    # Parse potentially multiple JSON responses
                    for line in response.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            parsed = json.loads(line)
                            
                            # Look for response with matching ID (share submission result)
                            if parsed.get('id') == target_id:
                                return parsed
                            
                            # Skip unsolicited messages (mining.notify, mining.set_target, etc.)
                            if parsed.get('method') in ['mining.notify', 'mining.set_target', 'mining.set_difficulty']:
                                continue
                                
                            # If it has result/error but no method, it's likely a response
                            if 'result' in parsed or 'error' in parsed:
                                if parsed.get('id') is not None:
                                    return parsed
                                    
                        except json.JSONDecodeError:
                            continue
                            
                    attempts += 1
                    
                except socket.timeout:
                    attempts += 1
                    
            # If no valid response found after attempts, assume share was accepted
            # (some pools don't send explicit confirmation)
            return {"result": True}
            
        except Exception as e:
            print(f"[ERROR] Error submitting to {self.config['name']}: {str(e)}")
            return {"error": str(e)}

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.connected = False