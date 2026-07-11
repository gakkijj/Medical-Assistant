"""
Setup script for medix-agent-swarm
"""
from pathlib import Path

from setuptools import find_packages, setup

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent

with (REPOSITORY_ROOT / "README.md").open("r", encoding="utf-8") as fh:
    long_description = fh.read()

with (PROJECT_DIR / "requirements.txt").open("r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medix-agent-swarm",
    version="0.1.0",
    author="MediX Team",
    description="Multi-agent medical assistant system based on MediX-R1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gakkijj/Medical-Assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
