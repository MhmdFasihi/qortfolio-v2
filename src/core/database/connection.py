"""MongoDB Database Connection Module"""

import os
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """MongoDB connection handler with singleton pattern"""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize database connection"""
        if self._client is None:
            self.connect()
    
    def connect(self) -> Optional[MongoClient]:
        """Establish MongoDB connection"""
        try:
            # Get connection string from environment or use default
            mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            
            # Create client with timeout
            self._client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            
            # Test connection
            self._client.admin.command('ping')
            logger.info("MongoDB connection established")
            return self._client
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB connection failed: {e}")
            self._client = None
            return None
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self._client = None
            return None
    
    @property
    def client(self) -> Optional[MongoClient]:
        """Get MongoDB client"""
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
            pass
        return False
    
    def get_collection(self, collection_name: str):
        """Get a specific collection"""
        if self.db:
            return self.db[collection_name]
        return None

# Create singleton instance
db_connection = DatabaseConnection()
