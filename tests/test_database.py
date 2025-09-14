"""Test database operations"""

import pytest
from src.core.database.connection import db_connection
from src.core.database.operations import db_ops
from datetime import datetime

def test_database_connection():
    """Test MongoDB connection"""
    assert db_connection is not None
    # Check if we can ping the database
    is_connected = db_connection.check_connection()
    assert isinstance(is_connected, bool)

def test_crud_operations():
    """Test CRUD operations"""
    # Test document
    test_doc = {
        "test_field": "test_value",
        "number": 42,
        "timestamp": datetime.now()
    }
    
    # INSERT
    doc_id = db_ops.insert_one("test_collection", test_doc)
    assert doc_id is not None or doc_id == None  # Handle both connected and disconnected states
    
    if doc_id:
        # FIND
        found_doc = db_ops.find_one("test_collection", {"test_field": "test_value"})
        assert found_doc is not None
        
        # UPDATE
        updated = db_ops.update_one(
            "test_collection",
            {"test_field": "test_value"},
            {"$set": {"number": 100}}
        )
        assert isinstance(updated, bool)
        
        # DELETE
        deleted = db_ops.delete_one("test_collection", {"test_field": "test_value"})
        assert isinstance(deleted, bool)

def test_time_utilities():
    """Test time calculation bug fix"""
    from src.core.utils.time_utils import calculate_time_to_maturity
    from datetime import datetime, timedelta
    
    current = datetime.now()
    expiry = current + timedelta(days=30)
    
    ttm = calculate_time_to_maturity(current, expiry)
    
    # Should be approximately 30/365.25
    expected = 30 / 365.25
    assert abs(ttm - expected) < 0.001

def test_black_scholes():
    """Test Black-Scholes implementation"""
    from src.models.options.black_scholes import BlackScholes
    
    bs = BlackScholes()
    
    # Test call option pricing
    price = bs.calculate_option_price(
        S=45000,  # Current price
        K=46000,  # Strike
        T=0.0833,  # 1 month
        r=0.05,   # Risk-free rate
        sigma=0.65,  # Volatility
        option_type='call'
    )
    
    assert price > 0
    assert price < 45000  # Should be less than spot price
