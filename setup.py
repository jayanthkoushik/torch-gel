"""setup.py: setup script for the project."""

import re
import os
from setuptools import setup, find_packages


# Get the version number
version_file = os.path.join(os.path.dirname(__file__), "gel", "_version.py")
with open(version_file) as f:
    version_str = f.read()
version_str_pattern = r'^__version__ = "(.*?)"$'
version = re.search(version_str_pattern, version_str, re.M).group(1)


setup(
    name="torchgel",
    version=version,
    description="PyTorch implementation of group elastic net",
    url="https://github.com/jayanthkoushik/torch-gel",
    author="Jayanth Koushik",
    author_email="jnkoushik@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    keywords="pytorch lasso elasticnet",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "tqdm>=4.0"
    ],
    python_requires=">=3.5",
    extras_require={
        "test": [
            "cvxpy>=0.4",
            "cvxopt"
        ]
    }
)
