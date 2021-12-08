"""Provides a consistent way to access the provided Dinwoodie and IEA Task 26 data libraries."""

import os
import re
import yaml  # type: ignore
from typing import Any, Union
from pathlib import Path


ROOT = Path(__file__).parents[2].resolve()
DEFAULT_LIBRARY = ROOT / "library"
DINWOODIE = DEFAULT_LIBRARY / "dinwoodie"
IEA_26 = DEFAULT_LIBRARY / "iea26"

library_map = {"DINWOODIE": DINWOODIE, "IEA_26": IEA_26}

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


def load_yaml(path: Union[str, Path], fname: str) -> Any:
    """Loads and returns the contents of the YAML file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the file to be loaded.
    fname : str
        Name of the file (ending in .yaml) to be loaded.

    Returns
    -------
    Any
        Whatever content is in the YAML file.
    """
    return yaml.load(open(os.path.join(path, fname), "r"), Loader=custom_loader)
