#!/usr/bin/env python3
"""Optional MongoDB connectivity test.

This test should never break the suite if MongoDB is not available or creds
are not configured. It will be skipped in that case.
"""

import os
import pytest
from pymongo import MongoClient


def _build_mongo_uri() -> str:
    # Prefer full URI if provided
    uri = os.getenv("MONGODB_URL")
    if uri:
        return uri

    host = os.getenv("MONGO_HOST", "localhost")
    port = os.getenv("MONGO_PORT", "27017")
    db = os.getenv("MONGO_DATABASE", "qortfolio")
    user = os.getenv("MONGO_USER", "")
    pwd = os.getenv("MONGO_PASSWORD", "")
    auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin")

    if user and pwd:
        return f"mongodb://{user}:{pwd}@{host}:{port}/{db}?authSource={auth_source}"
    return f"mongodb://{host}:{port}/{db}"


def test_mongodb_connection():
    uri = _build_mongo_uri()
    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        # Ping server
        client.admin.command("ping")
        # Access DB and basic sanity
        db_name = os.getenv("MONGO_DATABASE", "qortfolio")
        db = client[db_name]
        _ = db.list_collection_names()
        assert True
    except Exception as e:
        print(f"⚠️ MongoDB not available for tests: {e}")
        pytest.skip("MongoDB not running or auth failed; skipping optional DB test")
    finally:
        try:
            if client:
                client.close()
        except Exception:
            pass
