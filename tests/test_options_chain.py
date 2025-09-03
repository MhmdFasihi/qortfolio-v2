#!/usr/bin/env python3
"""
Test Options Chain Processor
File: tests/test_options_chain.py
Run: python tests/test_options_chain.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
import json

def test_options_chain_processor():
    """Test Options Chain Processor with Deribit data."""
    
    print("=" * 60)
    print("Testing Options Chain Processor")
    print("=" * 60)
    
    try:
        # Import the modules
        from src.models.options.options_chain import (
            OptionsChainProcessor, OptionChainMetrics, OptionContract,
            process_deribit_options, analyze_options_chain
        )
        from src.models.options.black_scholes import BlackScholesModel
        from src.models.options.greeks_calculator import GreeksCalculator
        
        print("✅ Options Chain module imported successfully")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure:")
        print("  - src/models/options/options_chain.py exists")
        print("  - src/models/options/black_scholes.py exists")
        print("  - src/models/options/greeks_calculator.py exists")
        return False
    
    # Test 1: Initialize processor
    print("\n1. Testing processor initialization...")
    try:
        bs_model = BlackScholesModel()
        greeks_calc = GreeksCalculator(bs_model)
        processor = OptionsChainProcessor(bs_model, greeks_calc)
        print("   ✅ Processor initialized with models")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False
    
    # Test 2: Create sample Deribit data
    print("\n2. Creating sample Deribit options data...")
    try:
        # Sample Deribit options (similar to what you collected)
        sample_deribit_data = [
            {
                'instrument_name': 'BTC-28JUN24-50000-C',
                'underlying_price': 50000,
                'index_price': 50000,
                'mark_price': 0.0523,  # In BTC
                'mark_iv': 80,  # 80% IV
                'best_bid_price': 0.0520,
                'best_ask_price': 0.0526,
                'volume': 125.5,
                'open_interest': 523.2
            },
            {
                'instrument_name': 'BTC-28JUN24-52000-C',
                'underlying_price': 50000,
                'index_price': 50000,
                'mark_price': 0.0385,  # In BTC
                'mark_iv': 82,  # 82% IV
                'best_bid_price': 0.0380,
                'best_ask_price': 0.0390,
                'volume': 89.3,
                'open_interest': 412.7
            },
            {
                'instrument_name': 'BTC-28JUN24-48000-P',
                'underlying_price': 50000,
                'index_price': 50000,
                'mark_price': 0.0412,  # In BTC
                'mark_iv': 85,  # 85% IV
                'best_bid_price': 0.0408,
                'best_ask_price': 0.0416,
                'volume': 156.8,
                'open_interest': 687.3
            },
            {
                'instrument_name': 'BTC-28JUN24-50000-P',
                'underlying_price': 50000,
                'index_price': 50000,
                'mark_price': 0.0465,
                'mark_iv': 80,
                'best_bid_price': 0.0460,
                'best_ask_price': 0.0470,
                'volume': 201.3,
                'open_interest': 892.1
            },
            {
                'instrument_name': 'BTC-15JUL24-55000-C',
                'underlying_price': 50000,
                'index_price': 50000,
                'mark_price': 0.0612,
                'mark_iv': 78,
                'best_bid_price': 0.0608,
                'best_ask_price': 0.0616,
                'volume': 67.2,
                'open_interest': 312.4
            }
        ]
        
        print(f"   ✅ Created {len(sample_deribit_data)} sample options")
        
    except Exception as e:
        print(f"   ❌ Sample data creation failed: {e}")
        return False
    
    # Test 3: Process Deribit chain
    print("\n3. Testing Deribit chain processing...")
    try:
        chain_df = processor.process_deribit_chain(sample_deribit_data)
        
        print(f"   ✅ Processed {len(chain_df)} options")
        print(f"   Columns: {', '.join(chain_df.columns[:8])}")
        
        # Display sample processed data
        print("\n   Sample processed options:")
        for i, row in chain_df.head(3).iterrows():
            print(f"     {row['instrument_name']}:")
            print(f"       Strike: ${row['strike']:,.0f}, Type: {row['option_type']}")
            print(f"       Mark: {row['mark_price']:.4f} BTC, IV: {row['implied_volatility']*100:.1f}%")
            if pd.notna(row['delta']):
                print(f"       Delta: {row['delta']:.6f}, Gamma: {row['gamma']:.9f}")
            print(f"       Moneyness: {row['moneyness']:.3f}, ATM: {row['is_atm']}")
        
    except Exception as e:
        print(f"   ❌ Chain processing failed: {e}")
        return False
    
    # Test 4: Analyze chain metrics
    print("\n4. Testing chain metrics analysis...")
    try:
        metrics = processor.analyze_chain_metrics(chain_df)
        
        print(f"   ✅ Chain Metrics:")
        print(f"     Total Volume: {metrics.total_volume:.2f}")
        print(f"     Total Open Interest: {metrics.total_open_interest:.2f}")
        print(f"     Put/Call Ratio: {metrics.put_call_ratio:.3f}")
        print(f"     Average IV: {metrics.average_iv*100:.1f}%")
        print(f"     ATM IV: {metrics.atm_iv*100:.1f}%")
        print(f"     IV Skew: {metrics.iv_skew*100:.2f}%")
        print(f"     Max Pain Strike: ${metrics.max_pain_strike:,.0f}")
        print(f"     Gamma Max Strike: ${metrics.gamma_max_strike:,.0f}")
        print(f"     Total Gamma Exposure: ${metrics.total_gamma_exposure:,.2f}")
        
        # Check term structure
        if metrics.term_structure:
            print("\n     Term Structure:")
            for expiry, iv in metrics.term_structure.items():
                print(f"       {expiry[:10]}: {iv*100:.1f}%")
        
    except Exception as e:
        print(f"   ❌ Metrics analysis failed: {e}")
        return False
    
    # Test 5: Identify opportunities
    print("\n5. Testing opportunity identification...")
    try:
        opportunities = processor.identify_opportunities(chain_df)
        
        print(f"   ✅ Opportunities found:")
        for opp_type, opps in opportunities.items():
            if opps:
                print(f"     {opp_type}: {len(opps)} opportunities")
                # Show first opportunity
                if opps:
                    print(f"       Example: {opps[0]}")
        
        if not any(opportunities.values()):
            print("     No specific opportunities identified (normal for small sample)")
        
    except Exception as e:
        print(f"   ❌ Opportunity identification failed: {e}")
        return False
    
    # Test 6: Quick functions
    print("\n6. Testing quick processing functions...")
    try:
        # Quick process
        quick_df = process_deribit_options(sample_deribit_data)
        print(f"   ✅ Quick process: {len(quick_df)} options processed")
        
        # Quick analysis
        analysis = analyze_options_chain(quick_df)
        print(f"   ✅ Quick analysis completed")
        print(f"     Metrics keys: {', '.join(list(analysis['metrics'].keys())[:5])}")
        
    except Exception as e:
        print(f"   ❌ Quick functions failed: {e}")
        return False
    
    # Test 7: Edge cases
    print("\n7. Testing edge cases...")
    try:
        # Empty chain
        empty_df = processor.process_deribit_chain([])
        print(f"   ✅ Empty chain handled: {len(empty_df)} rows")
        
        # Invalid instrument name
        bad_data = [{
            'instrument_name': 'INVALID-FORMAT',
            'underlying_price': 50000,
            'mark_price': 0.05,
            'mark_iv': 80
        }]
        bad_df = processor.process_deribit_chain(bad_data)
        print(f"   ✅ Invalid data handled gracefully")
        
        # Missing fields
        incomplete_data = [{
            'instrument_name': 'BTC-28JUN24-50000-C',
            'underlying_price': 50000
            # Missing other fields
        }]
        incomplete_df = processor.process_deribit_chain(incomplete_data)
        print(f"   ✅ Incomplete data handled")
        
    except Exception as e:
        print(f"   ❌ Edge case testing failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All Options Chain tests passed!")
    print("=" * 60)
    return True


def test_deribit_live_integration():
    """Test with actual Deribit collector if available."""
    print("\n" + "=" * 60)
    print("Testing Live Deribit Integration")
    print("=" * 60)
    
    try:
        # Try to import Deribit collector
        from src.data.collectors.deribit_collector import DeribitCollector
        from src.models.options.options_chain import OptionsChainProcessor
        
        print("Attempting to fetch live Deribit data...")
        
        collector = DeribitCollector()
        processor = OptionsChainProcessor()
        
        # Try to get BTC options
        try:
            btc_options = collector.get_options_data('BTC')
            
            if btc_options:
                print(f"✅ Fetched {len(btc_options)} BTC options from Deribit")
                
                # Process the chain
                chain_df = processor.process_deribit_chain(btc_options)
                print(f"✅ Processed {len(chain_df)} options with Greeks")
                
                # Analyze
                metrics = processor.analyze_chain_metrics(chain_df)
                print(f"✅ Analysis complete:")
                print(f"   Average IV: {metrics.average_iv*100:.1f}%")
                print(f"   Put/Call Ratio: {metrics.put_call_ratio:.3f}")
                
                # Save sample to file for inspection
                sample_file = "tests/sample_processed_chain.csv"
                chain_df.head(20).to_csv(sample_file, index=False)
                print(f"✅ Sample saved to: {sample_file}")
                
            else:
                print("⚠️ No live data available (API might be down or need credentials)")
                
        except Exception as e:
            print(f"⚠️ Live fetch failed (expected if no API credentials): {e}")
            print("   This is normal for testing without live API access")
        
    except ImportError:
        print("ℹ️ Deribit collector not available - skipping live test")
        print("   This is normal if you haven't set up the collector yet")
    
    return True  # Don't fail the test suite for optional live integration


def test_mongodb_integration():
    """Test MongoDB integration if available."""
    print("\n" + "=" * 60)
    print("Testing MongoDB Integration")
    print("=" * 60)
    
    try:
        from src.core.database.operations import DatabaseOperations
        from src.models.options.options_chain import OptionsChainProcessor
        
        # Check if MongoDB is running
        db_ops = DatabaseOperations()
        
        if db_ops.db is not None:
            print("✅ MongoDB connection available")
            
            processor = OptionsChainProcessor(db_ops=db_ops)
            
            # Create sample data
            sample_data = [{
                'instrument_name': 'BTC-28JUN24-50000-C',
                'underlying_price': 50000,
                'mark_price': 0.0523,
                'mark_iv': 80,
                'volume': 100,
                'open_interest': 500
            }]
            
            chain_df = processor.process_deribit_chain(sample_data)
            metrics = processor.analyze_chain_metrics(chain_df)
            
            # Try to save
            success = processor.save_to_database(chain_df, metrics)
            
            if success:
                print("✅ Successfully saved to MongoDB")
            else:
                print("⚠️ MongoDB save failed (check connection)")
        else:
            print("ℹ️ MongoDB not connected - skipping database test")
            
    except ImportError:
        print("ℹ️ Database operations not available - skipping MongoDB test")
    except Exception as e:
        print(f"⚠️ MongoDB test failed: {e}")
        print("   This is normal if MongoDB is not running")
    
    return True  # Don't fail for optional MongoDB


if __name__ == "__main__":
    print("Options Chain Processor Test Suite")
    print("File: tests/test_options_chain.py")
    print("-" * 60)
    
    # Run main tests
    success = test_options_chain_processor()
    
    # Run optional integration tests
    if success:
        test_deribit_live_integration()
        test_mongodb_integration()
    
    if success:
        print("\n🎉 All Options Chain tests passed!")
        print("\n📊 Summary of completed components:")
        print("✅ Black-Scholes Model (src/models/options/black_scholes.py)")
        print("✅ Greeks Calculator (src/models/options/greeks_calculator.py)")
        print("✅ Options Chain Processor (src/models/options/options_chain.py)")
        print("\n🚀 Ready for integration with:")
        print("- Your Deribit collector (582 BTC options)")
        print("- MongoDB storage")
        print("- Reflex dashboard")
        print("\nNext steps:")
        print("1. Integrate with your live Deribit data")
        print("2. Store calculated Greeks in MongoDB")
        print("3. Create Reflex dashboard pages for options analytics")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("Common issues:")
        print("- Ensure all three model files exist in src/models/options/")
        print("- Check that imports are correct")
        print("- Verify Black-Scholes and Greeks Calculator work first")
    
    sys.exit(0 if success else 1)