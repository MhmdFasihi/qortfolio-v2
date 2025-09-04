# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
MongoDB connection management for Qortfolio V2.
Handles connection pooling and async operations.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import motor.motor_asyncio
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from ..config import config
from ..exceptions import DatabaseConnectionError, DatabaseOperationError

logger = logging.getLogger(__name__)

def _mask_mongo_uri(uri: str) -> str:
    """Mask credentials in a MongoDB URI for safe logging."""
    try:
        if "://" in uri and "@" in uri:
            scheme, rest = uri.split("://", 1)
            if "@" in rest:
                _, tail = rest.split("@", 1)
                return f"{scheme}://***@{tail}"
        return uri
    except Exception:
        return uri

class DatabaseConnection:
    """MongoDB connection manager with async support."""
    
    def __init__(self):
        self.sync_client: Optional[MongoClient] = None
        self.async_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db_name = config.database.database
        self._connected = False
        
    def connect_sync(self) -> MongoClient:
        """Establish synchronous MongoDB connection."""
        if self.sync_client is None:
            try:
                logger.info("Attempting MongoDB (sync) connection: %s", _mask_mongo_uri(config.database.connection_string))
                self.sync_client = MongoClient(
                    config.database.connection_string,
                    maxPoolSize=50,
                    minPoolSize=10,
                    serverSelectionTimeoutMS=5000
                )
                # Test connection
                self.sync_client.admin.command('ping')
                self._connected = True
                logger.info("✅ Synchronous MongoDB connection established")
            except (ConnectionFailure, OperationFailure) as e:
                # Ensure client is closed to stop background retries
                try:
                    if self.sync_client is not None:
                        self.sync_client.close()
                finally:
                    self.sync_client = None
                logger.error(f"❌ Failed to connect to MongoDB: {e}")
                raise DatabaseConnectionError(f"MongoDB connection failed: {e}")
        
        return self.sync_client
    
    async def connect_async(self) -> motor.motor_asyncio.AsyncIOMotorClient:
        """Establish asynchronous MongoDB connection."""
        if self.async_client is None:
            try:
                logger.info("Attempting MongoDB (async) connection: %s", _mask_mongo_uri(config.database.connection_string))
                self.async_client = motor.motor_asyncio.AsyncIOMotorClient(
                    config.database.connection_string,
                    maxPoolSize=50,
                    minPoolSize=10,
                    serverSelectionTimeoutMS=5000
                )
                # Test connection
                await self.async_client.admin.command('ping')
                self._connected = True
                logger.info("✅ Asynchronous MongoDB connection established")
            except Exception as e:
                # Ensure client is closed to stop background retries
                try:
                    if self.async_client is not None:
                        self.async_client.close()
                finally:
                    self.async_client = None
                logger.error(f"❌ Failed to connect to MongoDB: {e}")
                raise DatabaseConnectionError(f"MongoDB connection failed: {e}")
        
        return self.async_client
    
    def get_database(self):
        """Get database instance (sync)."""
        if not self.sync_client:
            self.connect_sync()
        return self.sync_client[self.db_name]
    
    async def get_database_async(self):
        """Get database instance (async)."""
        if not self.async_client:
            await self.connect_async()
        return self.async_client[self.db_name]
    
    def close(self):
        """Close all connections."""
        if self.sync_client:
            self.sync_client.close()
            self.sync_client = None
        if self.async_client:
            self.async_client.close()
            self.async_client = None
        self._connected = False
        logger.info("MongoDB connections closed")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to MongoDB."""
        return self._connected
    
    def check_connection(self) -> bool:
        """Backward-compatible connectivity check (boolean)."""
        try:
            if not self.sync_client:
                self.connect_sync()
            # Ping database
            self.sync_client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connection."""
        try:
            if not self.sync_client:
                self.connect_sync()
            
            # Ping database
            self.sync_client.admin.command('ping')
            
            # Get server info
            server_info = self.sync_client.server_info()
            
            # Get database stats
            db = self.get_database()
            stats = db.command("dbstats")
            
            return {
                "status": "healthy",
                "connected": True,
                "server_version": server_info.get("version"),
                "database": self.db_name,
                "collections": db.list_collection_names(),
                "size_mb": stats.get("dataSize", 0) / (1024 * 1024),
                "indexes": stats.get("indexes", 0)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }

# Global connection instance
db_connection = DatabaseConnection()

async def test_connection():
    """Test database connection."""
    try:
        # Test sync connection
        sync_client = db_connection.connect_sync()
        print("✅ Sync connection successful")
        
        # Test async connection
        async_client = await db_connection.connect_async()
        print("✅ Async connection successful")
        
        # Health check
        health = db_connection.health_check()
        print(f"📊 Database health: {health}")
        
        return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Run connection test
    asyncio.run(test_connection())
