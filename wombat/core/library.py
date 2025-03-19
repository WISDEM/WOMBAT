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

import re
from copy import deepcopy
from typing import Any
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2].resolve()
DEFAULT_LIBRARY = ROOT / "library"
CODE_COMPARISON = DEFAULT_LIBRARY / "code_comparison"
BASE_CASES = DEFAULT_LIBRARY / "baseline"

DINWOODIE = CODE_COMPARISON / "dinwoodie"
IEA_26 = CODE_COMPARISON / "iea26"
COREWIND = DEFAULT_LIBRARY / "corewind"

OSW_FIXED = BASE_CASES / "offshore_fixed"
LBW = BASE_CASES / "land_based"

library_map = {
    "DINWOODIE": DINWOODIE,
    "IEA_26": IEA_26,
    "IEA26": IEA_26,
    "COREWIND": COREWIND,
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
        Name of the file (ending in .yaml or .yml) to be loaded.

    Returns
    -------
    Any
        Whatever content is in the YAML file.
    """
    if isinstance(path, str):
        path = Path(path).resolve()
    with (path / fname).open() as f:
        return yaml.load(f, Loader=custom_loader)


def create_library_structure(
    library_path: str | Path, create_init: bool = False
) -> None:
    """Creates the following library structure at ``library_path``. If ``library_path``
    does not exist, then the method will fail.

    .. code-block:: text

        <library_path>/
        └── project
            └── config     <- Project-level configuration files
            └── port       <- Port configuration files
            └── plant      <- Wind farm layout files
        └── cables       <- Export and Array cable configuration files
        └── substations  <- Substation configuration files
        └── turbines     <- Turbine configuration and power curve files
        └── vessels      <- Land-based and offshore servicing equipment configuration files
        └── weather      <- Weather profiles
        └── results      <- The analysis log files and any saved output data

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
    """  # noqa: E501
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


def convert_failure_data(
    configuration: str | Path | dict,
    which: str,
    save_name: str | Path | None = None,
    return_dict: bool = False,
) -> None | dict:
    """Converts the pre-v0.10 failure configuration data for cable, turbine, substation
    data in both individual files or consolidated configurations to be in the v0.10+
    style.

    Parameters
    ----------
    configuration : str | Path | dict
        The configuration file or dictionary containing failure data.
    which : str
        The type of configuration. Muat be one of "cable", "substation", "turbine",
        or "configuration" where "configuration" is a consolidated simulation
        configuration file containing any or all of the different types.
    save_name : str | Path | None, optional
        The file path and name of where to save the converted configuration, by default
        None.
    return_dict : bool, optional
        Use True to return the converted dictionary, by default False.

    Returns
    -------
    None | dict
        If :py:attr:`return_dict` is True, then `dict`, otherwise None.

    Raises
    ------
    FileNotFoundError
        Raised if the :py:attr:`configuration` can't be found.
    ValueError
        Raised if :py:attr:`configuration` can't be converted to a dictionary because
        a dictionary was not passed nor was a valid file path to load.
    ValueError
        Raised if :py:attr:`which` received an invalid input.
    """
    configuration = deepcopy(configuration)
    original = deepcopy(configuration)
    if isinstance(configuration, str):
        configuration = Path(configuration)
    if isinstance(configuration, Path):
        configuration = configuration.resolve()
        if not configuration.is_file():
            msg = f"{configuration} cannot be found, please check the path."
            raise FileNotFoundError(msg)
        configuration = load_yaml(configuration.parent, configuration.name)

    if not isinstance(configuration, dict):
        if isinstance(original, (str, Path)):
            msg = f"{original} could not be converted to a dictionary."
        else:
            msg = "Input to `configuration` was not a dictionary."
        raise TypeError(msg)

    opts = ("cable", "turbine", "substation", "configuration")
    match which:
        case "cable":
            configuration["failures"] = list(configuration["failures"].values())
        case "turbine" | "substation":
            for key, val in configuration.items():
                if not isinstance(val, dict) or key == "power_curve":
                    continue
                configuration[key]["failures"] = list(
                    configuration[key]["failures"].values()
                )
        case "configuration":
            if "cables" in configuration:
                for name, config in configuration["cables"].items():
                    configuration["cables"][name]["failures"] = list(
                        config["failures"].values()
                    )
            if "turbines" in configuration:
                for name, config in configuration["turbines"].items():
                    for key, val in config.items():
                        if isinstance(val, dict) and key != "power_curve":
                            configuration["turbines"][name][key]["failures"] = list(
                                val["failures"].values()
                            )
            if "substations" in configuration:
                for name, config in configuration["substations"].items():
                    for key, val in config.items():
                        if isinstance(val, dict):
                            configuration["substations"][name][key]["failures"] = list(
                                val["failures"].values()
                            )
        case _:
            raise ValueError(f"`which` must be one of: {', '.join(opts)}")

    if save_name is not None:
        with Path(save_name).open("w") as f:
            yaml.dump(configuration, f, default_flow_style=False, sort_keys=False)

    if return_dict:
        return configuration
    return None
