"""The package setup for installations."""

from __future__ import annotations

import codecs
from pathlib import Path

from setuptools import setup, find_packages


HERE = Path(__file__).parent

with open(HERE / "README.md", "r", newline="") as readme:
    long_description = readme.read()
    long_description_content_type = "text/markdown"


def read(relative_path: Path | str) -> str:
    """Reads the `relative_path` file."""
    with codecs.open(str(HERE / relative_path), "r") as fp:
        return fp.read()


def get_version(relative_path: Path | str) -> str:
    """Retreives the version number."""
    for line in read(relative_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


description = "Windfarm operations and maintenance cost-benefit analysis tool"
version = get_version(HERE / "wombat" / "__init__.py")


# Requirements
REQUIRED = [
    "attrs>=21",
    "numpy>=1.21",
    "scipy>=1.8",
    "pandas>=1.4",
    "pyarrow>=10",
    "jupyterlab>=3",
    "simpy>=4.0.1",
    "pyyaml>=6",
    "geopy>=2.3",
    "networkx>=2.7",
    "matplotlib>=3.3",
    "nrel-pysam>=4",
    # required for pre-commit CI
    "types-attrs>=19",
    "types-typed-ast>=1.5",
    "types-PyYAML>=6",
    "types-python-dateutil>=2.8",
]
DEVELOPER = [
    "pre-commit>=2.20",
    "pylint>=2.14",
    "flake8>=5",
    "flake8-docstrings>=1.6",
    "flake8_sphinx_links==0.2.1",
    "black>=22.1",
    "isort>=5.10",
    "pytest>=7",
    "pytest-cov>=4",
    "mypy>=0.991",
]
DOCUMENTATION = [
    "Sphinx==4.*",
    "myst-nb>=0.16",
    "myst-parser>=0.17",
    "sphinx-panels>=0.6",
    "sphinx-book-theme>=0.3.3",
    "sphinxcontrib-spelling>=7",
    "linkify-it-py>=2",
    "sphinxcontrib-bibtex>=2.4",
]
extra_package_requirements = {
    "dev": DEVELOPER,
    "docs": DOCUMENTATION,
    "all": DEVELOPER + DOCUMENTATION,
}


setup(
    name="wombat",
    author="Rob Hammond",
    author_email="rob.hammond@nrel.gov",
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    project_urls={
        "Source": "https://github.com/WISDEM/WOMBAT",
        "Documentation": "https://wisdem.github.io/WOMBAT/",
    },
    classifiers=[
        # Source: https://pypi.org/classifiers
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
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={"": ["*.yaml", "*.csv"]},
    install_requires=REQUIRED,
    python_requires=">=3.8, <3.11",
    extras_require=extra_package_requirements,
    test_suite="pytest",
    tests_require=["pytest", "pytest-cov"],
)
