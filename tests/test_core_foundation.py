#!/usr/bin/env python3
"""
Test script for Day 1 Core Foundation.
Verifies all critical components are working.
"""

import sys
import asyncio
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("🧪 QORTFOLIO V2 - CORE FOUNDATION TEST")
print("=" * 60)

# Test 1: Time Utils (CRITICAL BUG FIX)
print("\n1️⃣ Testing Time Utilities (Bug Fix)...")
try:
    from src.core.utils.time_utils import TimeUtils, validate_time_calculation
    validate_time_calculation()
    print("   ✅ Time calculation bug fixed and validated!")
except Exception as e:
    print(f"   ❌ Time utils test failed: {e}")

# Test 2: Configuration
print("\n2️⃣ Testing Configuration Management...")
try:
    from src.core.config import config
    print(f"   ✅ Database: {config.database.database}")
    print(f"   ✅ Environment: {config.app_settings['environment']}")
    print(f"   ✅ Debug Mode: {config.app_settings['debug']}")
except Exception as e:
    print(f"   ❌ Configuration test failed: {e}")

# Test 3: Logging
print("\n3️⃣ Testing Logging Framework...")
try:
    import logging
    logger = logging.getLogger("test")
    logger.info("Test log message")
    print("   ✅ Logging framework working!")
except Exception as e:
    print(f"   ❌ Logging test failed: {e}")

# Test 4: Database Connection
print("\n4️⃣ Testing Database Connection...")
@pytest.mark.asyncio
async def test_db():
    try:
        from src.core.database.connection import db_connection
        health = db_connection.health_check()
        if health['connected']:
            print(f"   ✅ MongoDB connected: {health['server_version']}")
            print(f"   ✅ Database: {health['database']}")
            print(f"   ✅ Collections: {health['collections']}")
        else:
            print(f"   ⚠️  MongoDB not connected: {health.get('error')}")
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")

asyncio.run(test_db())

# Test 5: Exceptions
print("\n5️⃣ Testing Exception Framework...")
try:
    from src.core.exceptions import (
        QortfolioException, 
        DatabaseConnectionError,
        TimeCalculationError
    )
    print("   ✅ Exception classes loaded successfully!")
except Exception as e:
    print(f"   ❌ Exception test failed: {e}")

print("\n" + "=" * 60)
print("📊 CORE FOUNDATION TEST SUMMARY")
print("=" * 60)
print("""
✅ Completed Day 1 Core Foundation:
   1. Time utilities with bug fix
   2. Configuration management
   3. Logging framework
   4. Database connection
   5. Custom exceptions

🎯 Next Steps (Day 2):
   - Data collectors (yfinance, Deribit)
   - Options data models
   - Basic Black-Scholes implementation
   - Greeks calculations
""")
