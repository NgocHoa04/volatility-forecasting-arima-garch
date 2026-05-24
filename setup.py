"""
Setup configuration file for development installation.
"""

from setuptools import setup, find_packages

setup(
    name="volatility-forecasting",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
