"""Package initialization."""

import importlib.metadata

from wombat.core import Metrics, Simulation
from wombat.core.library import create_library_structure


__version__ = importlib.metadata.version("wombat")
