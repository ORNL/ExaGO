"""Package setup for AgentiGrid."""

from pathlib import Path

from setuptools import find_packages, setup

readme = Path(__file__).parent / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="agentigrid",
    version="0.1.0",
    description="LLM-driven iterative simulation and analysis tool for ExaGO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Slaven Peles",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "pyyaml>=6.0,<7",
        "openai>=1.0,<2",
        "anthropic>=0.30,<1",
        "ollama>=0.3,<1",
    ],
    entry_points={
        "console_scripts": [
            "agentigrid=agentigrid.cli:main",
        ],
    },
)
