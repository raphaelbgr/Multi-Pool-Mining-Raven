#!/usr/bin/env python3

nonce = 4212448

print(f"Nonce: {nonce}")
print(f"Hex: 0x{nonce:x}")
print(f"Big-endian: {nonce:08x}")

# Little-endian calculation
byte0 = (nonce >> 0) & 0xFF
byte1 = (nonce >> 8) & 0xFF
byte2 = (nonce >> 16) & 0xFF
byte3 = (nonce >> 24) & 0xFF

print(f"Bytes: {byte3:02x} {byte2:02x} {byte1:02x} {byte0:02x}")
print(f"Little-endian: {byte0:02x}{byte1:02x}{byte2:02x}{byte3:02x}")

print("\nWhat our adapters actually produce:")
from adapters.factory import AdapterFactory
import json

with open("config.json") as f:
    config = json.load(f)

adapter = AdapterFactory.create_adapter(config['pools'][0])  # 2Miners
result = adapter._format_nonce(nonce)
print(f"Adapter result: {result}") 