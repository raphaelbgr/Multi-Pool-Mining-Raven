#!/usr/bin/env python3
"""
Script de diagnóstico para o minerador multi-pool CUDA
Verifica status das pools, jobs, minerador e shares
"""

import json
import os
import subprocess
import time
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinerDiagnostic:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        """Carrega configuração das pools"""
        try:
            with open("config.json") as f:
                config = json.load(f)
            logger.info(f"✅ Config loaded: {len(config['pools'])} pools")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load config: {str(e)}")
            return None
            
    def check_files(self):
        """Verifica se os arquivos necessários existem"""
        logger.info("📁 Checking required files...")
        
        required_files = [
            "config.json",
            "miner.exe",
            "get_jobs.py",
            "submit_share.py",
            "auto_miner.py"
        ]
        
        missing_files = []
        for file in required_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                logger.info(f"  ✅ {file} ({size} bytes)")
            else:
                logger.error(f"  ❌ {file} - MISSING")
                missing_files.append(file)
                
        if missing_files:
            logger.error(f"❌ Missing files: {missing_files}")
            return False
        return True
        
    def check_jobs(self):
        """Verifica status dos jobs atuais"""
        logger.info("📦 Checking current jobs...")
        
        if not os.path.exists("jobs.json"):
            logger.error("❌ jobs.json not found")
            return False
            
        if not os.path.exists("headers.bin"):
            logger.error("❌ headers.bin not found")
            return False
            
        try:
            with open("jobs.json") as f:
                jobs = json.load(f)
                
            logger.info(f"📊 Found {len(jobs)} jobs")
            
            active_jobs = 0
            for job in jobs:
                if job.get('header_hash'):
                    active_jobs += 1
                    logger.info(f"  ✅ Pool {job['pool_index']} ({job['pool_name']}): {job['job_id']}")
                else:
                    logger.warning(f"  ⚠️ Pool {job['pool_index']}: No job")
                    
            logger.info(f"📈 Active jobs: {active_jobs}/{len(jobs)}")
            
            # Verificar tamanho do headers.bin
            headers_size = os.path.getsize("headers.bin")
            expected_size = len(self.config['pools']) * (4 + 32 + 32)  # header_len + header + target
            logger.info(f"📏 headers.bin: {headers_size} bytes (expected: ~{expected_size})")
            
            return active_jobs > 0
            
        except Exception as e:
            logger.error(f"❌ Error reading jobs: {str(e)}")
            return False
            
    def test_miner(self):
        """Testa o minerador com um nonce pequeno"""
        logger.info("⛏️ Testing miner...")
        
        try:
            result = subprocess.run(
                ["./miner.exe", "0"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error("❌ Miner test failed:")
                logger.error(result.stderr)
                return False
                
            logger.info("✅ Miner test completed")
            logger.debug("Miner output:")
            logger.debug(result.stdout)
            
            # Verificar se encontrou algum nonce válido
            if "Valid nonce found" in result.stdout:
                logger.info("🎯 Found valid nonce during test!")
            else:
                logger.info("⚠️ No valid nonces found during test (this is normal)")
                
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Miner test timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Miner test error: {str(e)}")
            return False
            
    def test_pool_connection(self, pool):
        """Testa conexão com uma pool específica"""
        logger.info(f"🔗 Testing connection to {pool['name']}...")
        
        try:
            result = subprocess.run(
                ["python", "get_jobs.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Failed to get jobs from {pool['name']}")
                logger.error(result.stderr)
                return False
                
            logger.info(f"✅ Successfully connected to {pool['name']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing {pool['name']}: {str(e)}")
            return False
            
    def check_logs(self):
        """Verifica logs recentes"""
        logger.info("📋 Checking recent logs...")
        
        log_files = ["miner.log", "get_jobs.log", "submit_share.log"]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                size = os.path.getsize(log_file)
                logger.info(f"  📄 {log_file}: {size} bytes")
                
                # Mostrar últimas linhas do log
                try:
                    with open(log_file) as f:
                        lines = f.readlines()
                        if lines:
                            logger.info(f"    Last line: {lines[-1].strip()}")
                except:
                    pass
            else:
                logger.warning(f"  ⚠️ {log_file}: Not found")
                
    def run_full_diagnostic(self):
        """Executa diagnóstico completo"""
        logger.info("🔍 Starting full diagnostic...")
        logger.info(f"🕐 Time: {datetime.now()}")
        
        # Verificar arquivos
        if not self.check_files():
            logger.error("❌ File check failed")
            return False
            
        # Verificar jobs
        if not self.check_jobs():
            logger.warning("⚠️ Job check failed - try running get_jobs.py")
            
        # Testar minerador
        if not self.test_miner():
            logger.error("❌ Miner test failed")
            return False
            
        # Verificar logs
        self.check_logs()
        
        logger.info("✅ Diagnostic completed")
        return True
        
    def suggest_fixes(self):
        """Sugere correções baseadas no diagnóstico"""
        logger.info("💡 Suggestions:")
        
        if not os.path.exists("jobs.json"):
            logger.info("  🔧 Run: python get_jobs.py")
            
        if not os.path.exists("miner.exe"):
            logger.info("  🔧 Compile: nvcc miner.cu -o miner.exe")
            
        logger.info("  🔧 For detailed logs, check: miner.log, get_jobs.log, submit_share.log")
        logger.info("  🔧 To start mining: python auto_miner.py")

if __name__ == "__main__":
    diagnostic = MinerDiagnostic()
    
    if diagnostic.config:
        diagnostic.run_full_diagnostic()
        diagnostic.suggest_fixes()
    else:
        logger.error("❌ Cannot run diagnostic without config") 