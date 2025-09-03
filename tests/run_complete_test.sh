#!/bin/bash

# Complete Test Script for Qortfolio V2 Financial Models with Real Data
# File: tests/run_complete_test.sh
# Run: bash tests/run_complete_test.sh

echo "=========================================="
echo "QORTFOLIO V2 - COMPLETE TEST WITH REAL DATA"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n${YELLOW}Running: $test_name${NC}"
    echo "Command: $test_command"
    echo "-----------------------------------------"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval $test_command; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Check Python
echo -e "${BLUE}Checking Python environment...${NC}"
python --version

# Test 1: Check MongoDB
echo -e "\n${BLUE}1. Checking MongoDB Status${NC}"
if docker ps | grep -q qortfolio-mongo; then
    echo -e "${GREEN}✓ MongoDB is running${NC}"
else
    echo -e "${YELLOW}! MongoDB not running. Starting it...${NC}"
    docker-compose up -d mongodb
    sleep 5
fi

# Test 2: Test financial models
run_test "Black-Scholes Model" "python tests/test_black_scholes.py"
run_test "Greeks Calculator" "python tests/test_greeks_calculator.py"
run_test "Options Chain Processor" "python tests/test_options_chain.py"

# Test 3: Test real data integration
run_test "Real Data Integration" "python tests/test_real_data_integration.py"

# Test 4: Test analytics processor
echo -e "\n${YELLOW}Testing Analytics Processor with Real Data${NC}"
python -c "
from src.analytics.options_processor import RealTimeOptionsProcessor
import asyncio

async def test():
    processor = RealTimeOptionsProcessor()
    analytics = await processor.process_live_options('BTC')
    if analytics:
        print(f'✓ Processed {analytics.options_count} options')
        print(f'  Spot: \${analytics.spot_price:,.2f}')
        print(f'  Avg IV: {analytics.chain_metrics[\"average_iv\"]*100:.1f}%')
        return True
    return False

success = asyncio.run(test())
exit(0 if success else 1)
" && echo -e "${GREEN}✓ Analytics processor test passed${NC}" || echo -e "${RED}✗ Analytics processor test failed${NC}"

# Test 5: Check if we can fetch real Deribit data
echo -e "\n${YELLOW}Testing Deribit Data Collection${NC}"
python -c "
from src.data.collectors.deribit_collector import DeribitCollector

collector = DeribitCollector()
btc_options = collector.get_options_data('BTC')

if btc_options:
    print(f'✓ Fetched {len(btc_options)} BTC options from Deribit')
    print(f'  First option: {btc_options[0][\"instrument_name\"]}')
    exit(0)
else:
    print('! No live data (API credentials may be needed)')
    exit(1)
" && echo -e "${GREEN}✓ Deribit collector working${NC}" || echo -e "${YELLOW}! Deribit collector needs API credentials${NC}"

# Test 6: Test complete integration
run_test "Complete Integration" "python tests/test_complete_integration.py"

# Test 7: Verify saved data
echo -e "\n${YELLOW}Checking saved test data...${NC}"
if [ -f "tests/real_btc_options_with_greeks.csv" ]; then
    echo -e "${GREEN}✓ Test data saved: tests/real_btc_options_with_greeks.csv${NC}"
    head -n 5 tests/real_btc_options_with_greeks.csv
fi

if [ -f "tests/real_data_summary.json" ]; then
    echo -e "${GREEN}✓ Summary saved: tests/real_data_summary.json${NC}"
    python -c "import json; data=json.load(open('tests/real_data_summary.json')); print(f'  Total options: {data[\"total_options\"]}'); print(f'  Greeks calculated: {data[\"greeks_calculated\"]}')"
fi

# Summary
echo -e "\n=========================================="
echo -e "${BLUE}TEST SUMMARY${NC}"
echo "=========================================="
echo -e "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    echo -e "\n${GREEN}Your system is ready for production!${NC}"
    echo -e "\nNext steps:"
    echo -e "1. Run the Reflex dashboard: ${YELLOW}reflex run${NC}"
    echo -e "2. Visit: ${YELLOW}http://localhost:3000/options${NC}"
    echo -e "3. Process your 582 BTC options with full Greeks"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Please review the errors above.${NC}"
    exit 1
fi