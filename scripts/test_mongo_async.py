#!/usr/bin/env python3
"""Quick async test for MongoDB connectivity using project settings.

Runs a ping and prints basic server info using motor client.
"""

import asyncio
from src.core.database.connection import db_connection


async def main():
    try:
        client = await db_connection.connect_async()
        await client.admin.command('ping')
        info = await client.server_info()
        db = client[db_connection.db_name]
        collections = await db.list_collection_names()
        print("✅ Async MongoDB connection OK")
        print(f"Server version: {info.get('version')}")
        print(f"Database: {db_connection.db_name}")
        print(f"Collections: {collections}")
    except Exception as e:
        print(f"❌ Async MongoDB test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

