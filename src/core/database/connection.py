"""
MongoDB connection management for Qortfolio V2.
Handles connection pooling, async operations, and reconnection.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

import motor.motor_asyncio
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from ..settings import config
from ..exceptions import DatabaseConnectionError

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
    """MongoDB connection manager with async support and reconnection."""

    def __init__(self):
        self.sync_client: Optional[MongoClient] = None
        self.async_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db_name = config.database.database
        self._connected = False
        self._last_ping: float = 0.0
        self._ping_interval_sec: float = 30.0
        self._async_lock = asyncio.Lock()

    # === Sync connection ===
    def connect_sync(self) -> MongoClient:
        """Establish synchronous MongoDB connection."""
        if self.sync_client is None:
            try:
                logger.info("Attempting MongoDB (sync) connection: %s", _mask_mongo_uri(config.database.connection_string))
                self.sync_client = MongoClient(
                    config.database.connection_string,
                    maxPoolSize=config.database.max_pool_size,
                    minPoolSize=config.database.min_pool_size,
                    serverSelectionTimeoutMS=config.database.server_selection_timeout_ms,
                    connectTimeoutMS=config.database.connect_timeout_ms,
                    socketTimeoutMS=config.database.socket_timeout_ms,
                    waitQueueTimeoutMS=config.database.wait_queue_timeout_ms,
                    heartbeatFrequencyMS=config.database.heartbeat_frequency_ms,
                    retryWrites=True,
                )
                # Test connection
                self.sync_client.admin.command('ping')
                self._connected = True
                self._last_ping = time.time()
                logger.info("✅ Synchronous MongoDB connection established")
            except (ConnectionFailure, OperationFailure) as e:
                try:
                    if self.sync_client is not None:
                        self.sync_client.close()
                finally:
                    self.sync_client = None
                logger.error(f"❌ Failed to connect to MongoDB: {e}")
                raise DatabaseConnectionError(f"MongoDB connection failed: {e}")

        return self.sync_client

    # === Async connection ===
    async def connect_async(self) -> motor.motor_asyncio.AsyncIOMotorClient:
        """Establish asynchronous MongoDB connection."""
        if self.async_client is None:
            async with self._async_lock:
                if self.async_client is None:
                    try:
                        logger.info("Attempting MongoDB (async) connection: %s", _mask_mongo_uri(config.database.connection_string))
                        self.async_client = motor.motor_asyncio.AsyncIOMotorClient(
                            config.database.connection_string,
                            maxPoolSize=config.database.max_pool_size,
                            minPoolSize=config.database.min_pool_size,
                            serverSelectionTimeoutMS=config.database.server_selection_timeout_ms,
                            connectTimeoutMS=config.database.connect_timeout_ms,
                            socketTimeoutMS=config.database.socket_timeout_ms,
                            waitQueueTimeoutMS=config.database.wait_queue_timeout_ms,
                            heartbeatFrequencyMS=config.database.heartbeat_frequency_ms,
                            retryWrites=True,
                        )
                        # Test connection
                        await self.async_client.admin.command('ping')
                        self._connected = True
                        self._last_ping = time.time()
                        logger.info("✅ Asynchronous MongoDB connection established")
                    except Exception as e:
                        try:
                            if self.async_client is not None:
                                self.async_client.close()
                        finally:
                            self.async_client = None
                        logger.error(f"❌ Failed to connect to MongoDB: {e}")
                        raise DatabaseConnectionError(f"MongoDB connection failed: {e}")

        return self.async_client

    # === Get DB handles ===
    def get_database(self):
        """Get database instance (sync)."""
        self.ensure_connected_sync()
        return self.sync_client[self.db_name]

    async def get_database_async(self):
        """Get database instance (async)."""
        await self.ensure_connected_async()
        return self.async_client[self.db_name]

    # === Ensure + Reconnect ===
    def ensure_connected_sync(self):
        """Ensure sync client exists and is healthy. Reconnect if needed."""
        try:
            if self.sync_client is None:
                self.connect_sync()
                return
            now = time.time()
            if now - self._last_ping > self._ping_interval_sec:
                self.sync_client.admin.command('ping')
                self._last_ping = now
                self._connected = True
        except Exception:
            self.reconnect_sync()

    async def ensure_connected_async(self):
        """Ensure async client exists and is healthy. Reconnect if needed."""
        try:
            if self.async_client is None:
                await self.connect_async()
                return
            now = time.time()
            if now - self._last_ping > self._ping_interval_sec:
                await self.async_client.admin.command('ping')
                self._last_ping = now
                self._connected = True
        except Exception:
            await self.reconnect_async()

    def reconnect_sync(self, retries: int = 3, base_delay: float = 1.0):
        """Reconnect sync client with exponential backoff."""
        if self.sync_client:
            try:
                self.sync_client.close()
            except Exception:
                pass
            self.sync_client = None
        self._connected = False
        last_err = None
        for attempt in range(retries):
            try:
                self.connect_sync()
                return
            except Exception as e:
                last_err = e
                time.sleep(base_delay * (2 ** attempt))
        raise DatabaseConnectionError(f"Sync reconnect failed: {last_err}")

    async def reconnect_async(self, retries: int = 3, base_delay: float = 1.0):
        """Reconnect async client with exponential backoff."""
        if self.async_client:
            try:
                self.async_client.close()
            except Exception:
                pass
            self.async_client = None
        self._connected = False
        last_err = None
        for attempt in range(retries):
            try:
                await self.connect_async()
                return
            except Exception as e:
                last_err = e
                await asyncio.sleep(base_delay * (2 ** attempt))
        raise DatabaseConnectionError(f"Async reconnect failed: {last_err}")

    # === Lifecycle ===
    def close(self):
        """Close all connections."""
        if self.sync_client:
            try:
                self.sync_client.close()
            except Exception:
                pass
            self.sync_client = None
        if self.async_client:
            try:
                self.async_client.close()
            except Exception:
                pass
            self.async_client = None
        self._connected = False
        logger.info("MongoDB connections closed")

    # === Health ===
    @property
    def is_connected(self) -> bool:
        return self._connected

    def check_connection(self) -> bool:
        try:
            if not self.sync_client:
                self.connect_sync()
            self.sync_client.admin.command('ping')
            return True
        except Exception:
            return False

    # === Backward-compatible helper used by simple CRUD layer ===
    def get_collection(self, collection_name: str):
        """Return a sync collection handle or None if unavailable."""
        try:
            db = self.get_database()
            return db[collection_name]
        except Exception:
            return None


# Global connection instance
db_connection = DatabaseConnection()
