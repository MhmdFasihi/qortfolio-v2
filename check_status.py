#!/usr/bin/env python3
"""Check status of all Qortfolio services."""

import subprocess
from pymongo import MongoClient
import redis
import sys

print("\n🔍 Qortfolio Infrastructure Status Check")
print("=" * 50)

# Check Docker containers
print("\n🐳 Docker Containers:")
result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}'], 
                       capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'qortfolio' in line.lower() or 'NAMES' in line:
        print(f"   {line}")

# Check MongoDB
print("\n💾 MongoDB:")
try:
    client = MongoClient('mongodb://admin:password123@localhost:27017/qortfolio?authSource=admin', 
                         serverSelectionTimeoutMS=2000)
    db = client.qortfolio
    
    # Server info
    server_info = client.server_info()
    print(f"   ✅ Connected (v{server_info['version']})")
    
    # Collections
    collections = db.list_collection_names()
    print(f"   Collections: {', '.join(collections)}")
    
    # Document counts
    for coll in collections:
        count = db[coll].count_documents({})
        if count > 0:
            print(f"   - {coll}: {count} documents")
    
    client.close()
except Exception as e:
    print(f"   ❌ Not connected: {e}")

# Check Redis (optional)
print("\n📦 Redis (Cache):")
try:
    r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2)
    r.ping()
    info = r.info()
    print(f"   ✅ Connected (v{info['redis_version']})")
    print(f"   Memory used: {info['used_memory_human']}")
except Exception:
    print("   ⚠️  Not running (optional)")

print("\n" + "=" * 50)
print("✅ Status check complete!")
