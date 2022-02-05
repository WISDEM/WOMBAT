"""The package setup for installations."""


import os
import codecs

from setuptools import setup, find_packages


def read(relative_path):
    """Reads the `relative_path` file."""
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, relative_path), "r") as fp:
        return fp.read()


def get_version(relative_path):
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
extra_package_requirements = {
    "dev": [
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
    ],
    "docs": [
        "sphinx",
        "myst-nb",
        "myst-parser",
        "sphinx-panels",
        "sphinx-book-theme",
        "sphinxcontrib-spelling",
        "linkify-it-py",
        "sphinxcontrib-bibtex",
    ],
}
extra_package_requirements["all"] = [
    *extra_package_requirements["dev"],
    *extra_package_requirements["docs"],
]


setup(
    name=name,
    author="Rob Hammond",
    author_email="rob.hammond@nrel.gov",
    version=get_version(os.path.join("wombat", "__init__.py")),
    description=description,
    long_description=long_description,
    project_urls={
        # "Documentation": "https://pip.pypa.io",
        "Source": "https://github.com/WISDEM/WOMBAT",
    },
    classifiers=[
        # TODO: https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 1 - Planning",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={"": ["*.yaml", "*.csv"]},
    install_requires=[
        "attrs>=21",
        "numpy>=1.21",
        "scipy",
        "pandas",
        "jupyterlab",
        "simpy>=4.0.1",
        "pyyaml",
        "geopy",
        "networkx",
        "matplotlib>=3.3",
        "nrel-pysam",
        # required for pre-commit CI
        "types-attrs",
        "types-PyYAML",
    ],
    python_requires=">=3.7",
    extras_require=extra_package_requirements,
    test_suite="pytest",
    tests_require=["pytest", "pytest-xdist", "pytest-cov"],
)
