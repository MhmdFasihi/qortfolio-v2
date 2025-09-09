"""MongoDB Database Connection Module"""

import os
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """MongoDB connection handler"""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self.connect()
    
    def connect(self) -> Optional[MongoClient]:
        """Establish MongoDB connection"""
        try:
            # Check for authentication credentials
            mongo_user = os.getenv('MONGO_USER', '')
            mongo_pass = os.getenv('MONGO_PASSWORD', '')
            mongo_host = os.getenv('MONGO_HOST', 'localhost')
            mongo_port = os.getenv('MONGO_PORT', '27017')
            
            # Build connection string
            if mongo_user and mongo_pass:
                mongo_uri = f"mongodb://{mongo_user}:{mongo_pass}@{mongo_host}:{mongo_port}/"
            else:
                mongo_uri = f"mongodb://{mongo_host}:{mongo_port}/"
            
            # Create client
            self._client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                directConnection=True
            )
            
            # Test connection with simple ping
            self._client.admin.command('ping')
            logger.info("MongoDB connection established")
            return self._client
            
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self._client = None
            return None
    
    @property
    def client(self) -> Optional[MongoClient]:
        if self._client is None:
            self.connect()
        return self._client
    
    @property
    def db(self):
        """Get default database"""
        if self.client:
            return self.client['qortfolio_v2']
        return None
    
    def check_connection(self) -> bool:
        """Check if connection is active"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
        except:
            self._client = None
        return False
    
    def get_collection(self, collection_name: str):
        """Get a specific collection"""
        if self.db:
            return self.db[collection_name]
        return None

db_connection = DatabaseConnection()
