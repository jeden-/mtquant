"""
MTQuant Trading System Setup

Production-grade multi-agent AI trading system using Reinforcement Learning.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mtquant",
    version="0.1.0",
    author="MTQuant Team",
    author_email="contact@mtquant.com",
    description="Multi-Agent AI Trading System using Reinforcement Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/jeden-/mtquant",
    project_urls={
        "Bug Reports": "https://github.com/jeden-/mtquant/issues",
        "Source": "https://github.com/jeden-/mtquant",
        "Documentation": "https://mtquant.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "ruff>=0.1.6",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
        ],
        "brokers": [
            "oandapyV20>=0.6.3",
            "ibapi>=10.19.1",
            "alpaca-trade-api>=3.1.1",
        ],
        "ml": [
            "optuna>=3.4.0",
            "wandb>=0.16.0",
            "mlflow>=2.8.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "mtquant=mtquant.cli:main",
            "mtquant-train=mtquant.agents.training.train:main",
            "mtquant-paper=scripts.paper_trade:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mtquant": [
            "config/*.yaml",
            "config/*.json",
            "*.md",
        ],
    },
    zip_safe=False,
    keywords="trading, reinforcement-learning, ai, finance, forex, crypto, mt4, mt5",
)
