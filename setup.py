"""The package setup for installations."""

from __future__ import annotations

import codecs
from pathlib import Path

from setuptools import setup, find_packages


def read(relative_path: Path | str) -> str:
    """Reads the `relative_path` file."""
    here = Path(__file__).resolve().parent
    with codecs.open(str(here / relative_path), "r") as fp:
        return fp.read()


def get_version(relative_path: Path | str) -> str:
    """Retreives the version number."""
    for line in read(relative_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.rst", "r") as fh:
    long_description = fh.read()

name = "wombat"
description = "Windfarm operations and maintenance cost-benefit analysis tool"

# Requirements
REQUIRED = [
    "attrs>=21",
    "numpy>=1.21",
    "scipy",
    "pandas>=1.3",
    "pyarrow>=10",
    "jupyterlab",
    "simpy>=4.0.1",
    "pyyaml",
    "geopy",
    "networkx",
    "matplotlib>=3.3",
    "nrel-pysam>=4",
    # required for pre-commit CI
    "types-attrs>=19",
    "types-typed-ast>=1.5",
    "types-PyYAML>=6",
    "types-python-dateutil>=2.8",
]
DEVELOPER = [
    "pre-commit",
    "pylint",
    "flake8",
    "flake8-docstrings",
    "black",
    "isort",
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "mypy",
]
DOCUMENTATION = [
    "sphinx",
    "myst-nb",
    "myst-parser",
    "sphinx-panels",
    "sphinx-book-theme",
    "sphinxcontrib-spelling",
    "linkify-it-py",
    "sphinxcontrib-bibtex",
]
extra_package_requirements = {
    "dev": DEVELOPER,
    "docs": DOCUMENTATION,
    "all": DEVELOPER + DOCUMENTATION,
}


setup(
    name=name,
    author="Rob Hammond",
    author_email="rob.hammond@nrel.gov",
    version=get_version(Path("wombat") / "__init__.py"),
    description=description,
    long_description=long_description,
    project_urls={
        "Source": "https://github.com/WISDEM/WOMBAT",
        "Documentation": "https://wisdem.github.io/WOMBAT/",
    },
    classifiers=[
        # TODO: https://pypi.org/classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topics :: Scientific/Engineering",
        "Topics :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={"": ["*.yaml", "*.csv"]},
    install_requires=REQUIRED,
    python_requires=">=3.8, <=3.10",
    extras_require=extra_package_requirements,
    test_suite="pytest",
    tests_require=["pytest", "pytest-xdist", "pytest-cov"],
)
