#!/usr/bin/env python3
"""Test MongoDB connection"""

from pymongo import MongoClient
import sys

try:
    # Connect to MongoDB
    client = MongoClient(
        'mongodb://admin:password123@localhost:27017/qortfolio?authSource=admin'
    )
    
    # Test connection
    db = client.qortfolio
    server_info = client.server_info()
    
    print("✅ MongoDB connection successful!")
    print(f"MongoDB version: {server_info['version']}")
    print(f"Database: qortfolio")
    
    # List collections
    collections = db.list_collection_names()
    print(f"Collections: {collections if collections else 'No collections yet'}")
    
    client.close()
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    sys.exit(1)
