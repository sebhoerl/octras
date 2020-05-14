#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = "octras",
    version = "1.0.0",
    description = "Optimization and Calibration for Transport Simulators",
    package_dir = { "": "src" },
    packages = find_packages(where = "src"),
    install_requires = [
        "GPy>=1.9.9",
        "emukit==0.4.7", #@https://github.com/Avidereta/emukit/tarball/my_contributions", #works with pip install (not python setup.py)
        "scipy==1.1.0", # Necessary for emukit
        "pandas>=0.25.3",
        "matplotlib>=3.1.2",
        "PyYAML>=5.1.2",
        "numpy",
        "deep_merge"
    ],
    extras_require = {
        "test": ["pytest>=5.3.1"]
    },
)
