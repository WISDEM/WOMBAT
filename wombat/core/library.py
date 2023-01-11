"""Provides a consistent way to access the provided Dinwoodie and IEA Task 26 data
libraries.

All library data should adhere to the followind directory structure where <library>
signifies the user's input library path:
```
<library>
  ├── project
    ├── config     <- Project-level configuration files
    ├── port       <- Port configuration files
    ├── plant      <- Wind farm layout files
  ├── cables       <- Export and Array cable configuration files
  ├── substations  <- Substation configuration files
  ├── turbines     <- Turbine configuration and power curve files
  ├── vessels      <- Land-based and offshore servicing equipment configuration files
  ├── weather      <- Weather profiles
  ├── results      <- The analysis log files and any saved output data
```
"""

from __future__ import annotations

import os
import re
from typing import Any
from pathlib import Path

import yaml


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


def create_library_structure(
    library_path: str | Path, create_init: bool = False
) -> None:
    """Creates the following library structure at ``library_path``. If ``library_path``
    does not exist, then the method will fail.

    ```
    <library_path>
    ├── project
        ├── config     <- Project-level configuration files
        ├── port       <- Port configuration files
        ├── plant      <- Wind farm layout files
    ├── cables       <- Export and Array cable configuration files
    ├── substations  <- Substation configuration files
    ├── turbines     <- Turbine configuration and power curve files
    ├── vessels      <- Land-based and offshore servicing equipment configuration files
    ├── weather      <- Weather profiles
    ├── results      <- The analysis log files and any saved output data
    ```

    Parameters
    ----------
    library_path : str | Path
        The folder under which the subfolder structure should be created.
    create_init : bool
        If True, create "__init__.py" in each of the folders so that python installation
        processes will register the files, and if False, only create the folders, by
        default False.

    Raises
    ------
    FileNotFoundError
        Raised if ``library_path`` is not a directory
    """
    if isinstance(library_path, str):
        library_path = Path(library_path)
    library_path = library_path.resolve(strict=True)
    if create_init:
        (library_path / "__init__.py").touch()

    folders = (
        "project/config",
        "project/port",
        "project/plant",
        "cables",
        "substations",
        "turbines",
        "vessels",
        "weather",
        "results",
    )
    # Make the project/ subfolder structure once and make the rest without parents
    (library_path / folders[0]).mkdir(parents=True, exist_ok=True)
    for folder in folders:
        f = library_path / folder
        f.mkdir(exist_ok=True)
        if create_init:
            f = f / "__init__.py"
            f.touch()
