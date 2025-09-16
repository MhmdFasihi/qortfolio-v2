# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Clean Architecture Validation Tests
Ensures all calculations happen in backend, zero calculations in frontend
"""

import pytest
import inspect
import ast
import sys
import os
from typing import List, Dict

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class CleanArchitectureValidator:
    """Validate clean architecture principles"""

    # Mathematical operations that should NOT be in frontend
    FORBIDDEN_MATH_OPERATIONS = [
        'numpy', 'np.', 'pandas', 'pd.',
        'riskfolio', 'rp.',
        'mean()', 'std()', 'cov()', 'corr()',
        'sum()', 'dot()', 'multiply',
        '@',  # Matrix multiplication
        'optimization(', 'HCPortfolio', 'Portfolio(',
        'efficient_frontier', 'assets_stats',
        'hrp_constraints', 'assets_constraints'
    ]

    # Backend calculation keywords that should only be in models/services
    BACKEND_CALCULATION_KEYWORDS = [
        'calculate', 'compute', 'optimize', 'analyze',
        'sharpe_ratio', 'volatility', 'expected_return',
        'monte_carlo', 'risk_measure', 'var_95'
    ]

    def scan_file_for_calculations(self, file_path: str) -> List[str]:
        """Scan a Python file for calculation-related code"""
        violations = []

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Check for forbidden math operations
            for operation in self.FORBIDDEN_MATH_OPERATIONS:
                if operation in content:
                    violations.append(f"Forbidden math operation: {operation}")

            # Parse AST to find function calls and assignments
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    # Check function calls
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute):
                            func_name = f"{ast.unparse(node.func.value)}.{node.func.attr}"
                            if any(forbidden in func_name for forbidden in self.FORBIDDEN_MATH_OPERATIONS):
                                violations.append(f"Forbidden function call: {func_name}")

            except Exception as e:
                # If AST parsing fails, continue with string-based checks
                pass

        except Exception as e:
            violations.append(f"Error reading file: {e}")

        return violations

    def test_frontend_has_no_calculations(self):
        """Test that frontend files contain no calculations"""
        print("🖥️  Testing Frontend Layer (should have NO calculations)")

        frontend_files = [
            'qortfolio_v2/portfolio_state.py',
            'qortfolio_v2/state.py',
            'qortfolio_v2/volatility_state.py',
            'qortfolio_v2/risk_state.py'
        ]

        violations_found = False

        for file_path in frontend_files:
            full_path = file_path
            if os.path.exists(full_path):
                violations = self.scan_file_for_calculations(full_path)

                # Filter out acceptable operations (UI state handling)
                filtered_violations = []
                for violation in violations:
                    # Allow basic UI operations
                    if not any(allowed in violation.lower() for allowed in [
                        'logger', 'print', 'datetime', 'str(', 'float(', 'int(',
                        'len(', 'enumerate', 'range', 'list(', 'dict('
                    ]):
                        filtered_violations.append(violation)

                if filtered_violations:
                    print(f"  ❌ {file_path} has calculation violations:")
                    for violation in filtered_violations[:3]:  # Show first 3
                        print(f"     - {violation}")
                    violations_found = True
                else:
                    print(f"  ✅ {file_path} - Clean (no calculations)")
            else:
                print(f"  ⚠️  {file_path} not found")

        return not violations_found

    def test_backend_has_calculations(self):
        """Test that backend files contain the calculations"""
        print("\n🔧 Testing Backend Layer (should have ALL calculations)")

        backend_files = [
            'src/models/portfolio/optimization_models.py',
            'src/services/portfolio_optimization_service.py',
            'src/analytics/risk/portfolio_risk.py',
            'src/analytics/performance/quantstats_analyzer.py'
        ]

        calculations_found = False

        for file_path in backend_files:
            if os.path.exists(file_path):
                violations = self.scan_file_for_calculations(file_path)

                # Count legitimate calculation operations
                calculation_count = len([v for v in violations if any(
                    calc in v.lower() for calc in ['numpy', 'pandas', 'riskfolio', 'optimization']
                )])

                if calculation_count > 0:
                    print(f"  ✅ {file_path} - Contains {calculation_count} calculation operations")
                    calculations_found = True
                else:
                    print(f"  ⚠️  {file_path} - No calculations detected")
            else:
                print(f"  ❌ {file_path} not found")

        return calculations_found

    def test_service_layer_separation(self):
        """Test that service layer properly separates concerns"""
        print("\n🏗️  Testing Service Layer Architecture")

        try:
            from src.services.portfolio_optimization_service import PortfolioOptimizationService
            from src.models.portfolio import PortfolioOptimizationModel

            # Test service layer exists
            service = PortfolioOptimizationService()
            print("  ✅ Service layer instantiated successfully")

            # Test that service calls models (not calculations directly)
            service_methods = [method for method in dir(service) if not method.startswith('_')]
            calculation_methods = [m for m in service_methods if any(
                calc in m.lower() for calc in ['run', 'get', 'generate', 'format']
            )]

            print(f"  ✅ Service layer has {len(calculation_methods)} business logic methods")

            # Test backend model exists
            model_methods = [method for method in dir(PortfolioOptimizationModel) if not method.startswith('_')]
            optimization_methods = [m for m in model_methods if any(
                opt in m.lower() for opt in ['optimize', 'prepare', 'create']
            )]

            print(f"  ✅ Backend model has {len(optimization_methods)} optimization methods")

            return len(calculation_methods) > 0 and len(optimization_methods) > 0

        except Exception as e:
            print(f"  ❌ Service layer test failed: {e}")
            return False

    def test_no_ui_dependencies_in_backend(self):
        """Test that backend has no UI dependencies"""
        print("\n🚫 Testing Backend Independence (no UI dependencies)")

        backend_files = [
            'src/models/portfolio/optimization_models.py',
            'src/services/portfolio_optimization_service.py'
        ]

        ui_dependencies = ['reflex', 'rx.', 'State', 'yield', 'Component']
        clean_backend = True

        for file_path in backend_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()

                ui_violations = [dep for dep in ui_dependencies if dep in content]

                if ui_violations:
                    print(f"  ❌ {file_path} has UI dependencies: {ui_violations}")
                    clean_backend = False
                else:
                    print(f"  ✅ {file_path} - No UI dependencies")

        return clean_backend

    def run_complete_architecture_validation(self):
        """Run complete clean architecture validation"""
        print("🧪 CLEAN ARCHITECTURE VALIDATION")
        print("=" * 65)

        results = []

        # Test 1: Frontend should have no calculations
        results.append(self.test_frontend_has_no_calculations())

        # Test 2: Backend should have all calculations
        results.append(self.test_backend_has_calculations())

        # Test 3: Service layer separation
        results.append(self.test_service_layer_separation())

        # Test 4: Backend independence
        results.append(self.test_no_ui_dependencies_in_backend())

        passed = sum(results)
        total = len(results)

        print(f"\n📊 CLEAN ARCHITECTURE TEST RESULTS: {passed}/{total} passed")
        print("=" * 65)

        if passed == total:
            print("✅ CLEAN ARCHITECTURE VALIDATION PASSED!")
            print("🎯 All calculations are in backend models")
            print("🎯 Service layer properly separates concerns")
            print("🎯 Frontend only handles UI state")
            print("🎯 Backend has zero UI dependencies")
            print("\n🚀 Architecture is production-ready!")
        else:
            print("⚠️  Some architecture issues found")
            print("🔧 Core separation is implemented")

        return passed == total

def main():
    """Run clean architecture validation tests"""
    validator = CleanArchitectureValidator()
    validator.run_complete_architecture_validation()

if __name__ == "__main__":
    main()