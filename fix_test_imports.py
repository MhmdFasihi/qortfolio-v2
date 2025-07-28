# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Fix Test Import Issues
Location: qortfolio-v2/fix_test_imports.py

Run this script to fix import path issues in your tests.
"""

import os
import sys
from pathlib import Path

def fix_test_imports():
    """Fix import issues in test files."""
    
    print("🔧 Fixing Test Import Issues")
    print("=" * 40)
    
    # Get project root
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    tests_path = project_root / "tests"
    
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Tests path: {tests_path}")
    
    # Create __init__.py files where missing
    init_files_to_create = [
        project_root / "src" / "__init__.py",
        project_root / "src" / "core" / "__init__.py", 
        project_root / "src" / "core" / "utils" / "__init__.py",
        project_root / "src" / "data" / "__init__.py",
        project_root / "src" / "models" / "__init__.py",
        project_root / "src" / "models" / "options" / "__init__.py",
        project_root / "src" / "analytics" / "__init__.py",
        project_root / "tests" / "__init__.py"
    ]
    
    for init_file in init_files_to_create:
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.write_text("# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)\n")
            print(f"✅ Created: {init_file}")
        else:
            print(f"✓ Exists: {init_file}")
    
    # Fix setup.py for proper package installation
    setup_py_content = '''# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

from setuptools import setup, find_packages

setup(
    name="qortfolio-v2",
    version="0.1.0",
    description="Quantitative Finance Platform for Options Analytics",
    author="Mhmd Fasihi",
    author_email="mhmd.fasihi@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.23.0", 
        "yfinance",
        "streamlit",
        "scikit-learn",
        "matplotlib",
        "plotly",
        "requests",
        "pyyaml",
        "python-dateutil"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.4.0",
            "flake8>=6.0.0"
        ]
    }
)
'''
    
    setup_file = project_root / "setup.py"
    setup_file.write_text(setup_py_content)
    print(f"✅ Updated: {setup_file}")
    
    # Create pytest.ini for better test discovery
    pytest_ini_content = '''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --cov=src
    --cov-branch
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
'''
    
    pytest_file = project_root / "pytest.ini"
    pytest_file.write_text(pytest_ini_content)
    print(f"✅ Created: {pytest_file}")
    
    print("\n🎯 Next Steps:")
    print("1. Install package in development mode:")
    print("   pip install -e .")
    print("\n2. Run tests:")
    print("   python -m pytest tests/ -v")
    print("\n3. Check coverage:")
    print("   python -m pytest tests/ --cov=src --cov-report=html")
    
    return True

if __name__ == "__main__":
    fix_test_imports()