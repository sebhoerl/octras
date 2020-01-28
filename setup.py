#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = "octras",
    version = "1.0",
    description = "Optimization and Calibration for Transport Simulators",
    package_dir = { "": "src" },
    packages = find_packages(where = "src"),
    install_requires = [
        "GPy>=1.9.9",
        "emukit==1.4.1", # github version with the link in dependency_links
        "scipy==1.1.0", # Necessary for emukit
        "pandas>=0.25.3",
        "matplotlib>=3.1.2",
        "pyDOE>=0.3.8",
        "PyYAML>=5.1.2",
        "numpy"
    ],
    dependency_links=['https://github.com/Avidereta/emukit/tarball/my_contributions#egg=emukit-1.4.1'],
    extras_require = {
        "test": ["pytest>=5.3.1"], "example": ["pandas>=0.25.3"]
    },
)
