from setuptools import setup, find_packages

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    entry_points={
        "console_scripts": [
            "qortfolio=src.dashboard.main:main",
        ],
    },
)