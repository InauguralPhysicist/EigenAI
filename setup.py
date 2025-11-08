"""
Setup configuration for EigenAI package.

This file allows you to install the package with: pip install -e .
"""

from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eigenai",
    version="0.1.0",
    author="Jon McReynolds",
    author_email="mcreynolds.jon@gmail.com",
    description="A framework for measuring AI understanding through eigenstate detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/InauguralPhysicist/EigenAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "full": [
            "spacy>=3.0.0",
            "matplotlib>=3.5.0",
            "pandas>=1.5.0",
        ],
    },
)
