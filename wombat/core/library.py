"""Provides a consistent way to access the provided Dinwoodie and IEA Task 26 data libraries."""

from __future__ import annotations

import os
import re
from typing import Any
from pathlib import Path

import yaml  # type: ignore


ROOT = Path(__file__).parents[2].resolve()
DEFAULT_LIBRARY = ROOT / "library"
CODE_COMPARISON = DEFAULT_LIBRARY / "code_comparison"
BASE_CASES = DEFAULT_LIBRARY / "baseline"

DINWOODIE = CODE_COMPARISON / "dinwoodie"
IEA_26 = DEFAULT_LIBRARY / "code_comparison" / "iea26"

OSW_FIXED = BASE_CASES / "offshore_fixed"
LBW = BASE_CASES / "land_based"

library_map = {
    "DINWOODIE": DINWOODIE,
    "IEA_26": IEA_26,
    "IEA26": IEA_26,
    "OSW_FIXED": OSW_FIXED,
    "LBW": LBW,
    "LAND_BASED": LBW,
}

# YAML loader that is able to read scientific notation
custom_loader = yaml.SafeLoader
custom_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_yaml(path: str | Path, fname: str | Path) -> Any:
    """Loads and returns the contents of the YAML file.

    Parameters
    ----------
    path : str | Path
        Path to the file to be loaded.
    fname : str | Path
        Name of the file (ending in .yaml) to be loaded.

    Returns
    -------
    Any
        Whatever content is in the YAML file.
    """
    return yaml.load(open(os.path.join(path, fname), "r"), Loader=custom_loader)
