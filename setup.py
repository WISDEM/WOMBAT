import codecs
import os
from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


name = "wombat"
description = "Windfarm Operations and Maintenance cost-Benefit Analysis Tool"

setup(
    name=name,
    version=get_version(os.path.join("wombat", "__init__.py")),
    description=description,
    long_description=read("README.md"),
    author="Rob Hammond",
    author_email="rob.hammond@nrel.gov",
    project_urls={"Source": "https://github.com/WISDEM/WOMBAT"},
    download_url="https://pypi.org/project/wombat/",
    classifiers=[
        # TODO: https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Version Control :: Git",
    ],
    packages=find_packages(),
    install_requires=[
        "attr",
        "numpy",
        "scipy",
        "pandas",
        "simpy>=4.0.1",
        "pyyaml",
        "geopy",
        "networkx",
        "matplotlib",
        "nrel-pysam",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pylint",
            "flake8",
            "black",
            "isort",
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "mypy",
        ],
        "docs": ["mkdocs"],
    },
    test_suite="pytest",
    tests_require=["pytest", "pytest-cov"],
)
