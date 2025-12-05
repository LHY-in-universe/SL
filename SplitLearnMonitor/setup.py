"""
Setup script for splitlearn-monitor
"""
from setuptools import setup, find_packages

# Read version from __version__.py
version = {}
with open("src/splitlearn_monitor/__version__.py") as f:
    exec(f.read(), version)

setup(
    name="splitlearn-monitor",
    version=version["__version__"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
