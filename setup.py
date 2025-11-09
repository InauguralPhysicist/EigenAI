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
    version="1.0.0",
    author="Jon McReynolds",
    author_email="mcreynolds.jon@gmail.com",
    description="Measure AI understanding through geometric eigenstate detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/InauguralPhysicist/EigenAI",
    project_urls={
        "Bug Tracker": "https://github.com/InauguralPhysicist/EigenAI/issues",
        "Documentation": "https://github.com/InauguralPhysicist/EigenAI#readme",
        "Source Code": "https://github.com/InauguralPhysicist/EigenAI",
        "Changelog": "https://github.com/InauguralPhysicist/EigenAI/blob/main/CHANGELOG.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="ai understanding eigenstate detection nlp semantic-analysis geometry physics",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",  # NumPy 2.0+ requires Python>=3.9
        "spacy>=3.5.0",
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
