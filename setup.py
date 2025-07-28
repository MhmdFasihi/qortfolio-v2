# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
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
