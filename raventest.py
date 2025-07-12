import socket

pools = [
    ("stratum.woolypooly.com", 55555),
    ("ravencoin.woolypooly.com", 55555),
    ("stratum+ssl.woolypooly.com", 55556)
]

for host, port in pools:
    try:
        s = socket.create_connection((host, port), timeout=5)
        print(f"✅ Success: {host}:{port}")
        s.close()
    except Exception as e:
        print(f"❌ Failed: {host}:{port} - {str(e)}")