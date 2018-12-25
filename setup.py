"""setup.py: setup script for the project."""

import os
from setuptools import find_packages, setup

# Get the long description.
readme_file = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_file) as f:
    long_description = f.read()

setup(
    name="torchgel",
    version="0.10.0",
    description="PyTorch implementation of group elastic net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jayanthkoushik/torch-gel",
    author="Jayanth Koushik",
    author_email="jnkoushik@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="pytorch lasso elasticnet",
    packages=find_packages(exclude=["tests"]),
    install_requires=["tqdm>=4.0"],
    python_requires=">=3.5",
    extras_require={
        "test": ["numpy", "cvxpy>=0.4,<1.0", "cvxopt"],
        "dev": ["black", "pylint", "isort", "twine", "wheel", "bumpversion"],
    },
)
