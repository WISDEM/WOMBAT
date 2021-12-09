"""The main API for the ``wombat``."""

import attr
import pandas as pd
from typing import List, Union, Optional
from pathlib import Path
from simpy.events import Event  # type: ignore

from wombat.core import (
    Metrics,
    FromDictMixin,
    RepairManager,
    ServiceEquipment,
    WombatEnvironment,
)
from wombat.windfarm import Windfarm
from wombat.core.library import load_yaml, library_map


def _library_mapper(file_path: Union[str, Path]) -> Path:
    """Attempts to extract a default library path if one of "DINWOODIE" or "IEA_26"
    are passed, other returns ``file_path``.

    Parameters
    ----------
    file_path : Union[str, Path]
        Should be a valid file path, or one of "DINWOODIE" or "IEA_26" to indicate a
        provided library is being used.

    Returns
    -------
    Union[str, Path]
        The library path.
    """
    return Path(library_map.get(file_path, file_path)).resolve()  # type: ignore


@attr.s(frozen=True, auto_attribs=True)
class Configuration(FromDictMixin):
    """The ``Simulation`` configuration data class that provides all the necessary definitions.

    Parameters
    ----------
    name: str
        Name of the simulation. Used for logging files.
    library : str
        The data directory. See ``wombat.simulation.WombatEnvironment`` for more details.
    layout : str
        The windfarm layout file. See ``wombat.Windfarm`` for more details.
    service_equipment : Union[str, List[str]]
        The equpiment that will be used in the simulation. See
        ``wombat.core.ServiceEquipment`` for more details.
    weather : str
        The weather profile to be used. See ``wombat.simulation.WombatEnvironment``
        for more details.
    workday_start : int
        Starting hour for a typical work shift. Can be overridden by
        equipment-specific settings.
    workday_end : int
        Ending hour for a typical work shift. Can be overridden by
        equipment-specific settings.
    inflation_rate : float
        The annual inflation rate to be used for post-processing.
    fixed_costs : str
        The file name for the fixed costs assumptions.
    project_capacity : Union[int, float]
        The total capacity of the wind plant, in MW.
    start_year : int
        Start year of the simulation. The exact date will be determined by
        the first valid date of this year in ``weather``.
    end_year : int
        Final year of the simulation. The exact date will be determined by
        the last valid date of this year in ``weather``.
    SAM_settings : str
        The SAM settings file to be used for financial modeling, optional, by
        default None.
    """

    name: str
    library: Path = attr.ib(converter=_library_mapper)
    layout: str
    service_equipment: Union[str, List[str]]
    weather: Union[str, pd.DataFrame]
    workday_start: int
    workday_end: int
    inflation_rate: float
    fixed_costs: str
    project_capacity: Union[int, float]
    start_year: int = attr.ib(default=None)
    end_year: int = attr.ib(default=None)
    SAM_settings: str = attr.ib(default=None)

    def __attrs_post_init__(self):
        if isinstance(self.service_equipment, str):
            object.__setattr__(self, "service_equipment", [self.service_equipment])


@attr.s(auto_attribs=True)
class Simulation(FromDictMixin):
    """The primary API to interact with the simulation methodologies.

    Parameters
    ----------
    name: str
        Name of the simulation. Used for logging files.
    library_path : str
        The path to the main data library.
    config : Union[str, dict]
        The path to a configuration dictionary or the dictionary itself.
    """

    name: str = attr.ib(converter=str)
    library_path: Path = attr.ib(converter=_library_mapper)
    config: Configuration = attr.ib()

    def __attrs_post_init__(self) -> None:
        self._setup_simulation()

    @config.validator  # type: ignore
    def _create_configuration(
        self, attribute: attr.Attribute, value: Union[str, dict, Configuration]
    ) -> None:
        """Validates the configuration object and creates the ``Configuration`` object
        for the simulation.Raises:

        Raises
        ------
        TypeError
            Raised if the value provided is not able to create a valid ``Configuration``
            object
        ValueError
            Raised if ``name`` and ``config.name`` or ``library_path`` and ``config.library``
            are not aligned.

        Returns
        -------
        Configuration
            The validated simulation configuration
        """
        if isinstance(value, str):
            value = load_yaml(self.library_path / "config", value)
        if isinstance(value, dict):
            value = Configuration.from_dict(value)
        if isinstance(value, Configuration):
            object.__setattr__(self, attribute.name, value)
        else:
            raise TypeError(
                "``config`` must be a dictionary, valid file path to a yaml-enocoded",
                "dictionary, or ``Configuration`` object!",
            )

        if self.config.name != self.name:
            raise ValueError("``name`` and the name in ``config`` do not match!")
        if self.config.library != self.library_path:
            raise ValueError(
                "``library_path`` and the library in ``config`` do not match!"
            )

    @classmethod
    def from_config(cls, config: Union[str, dict, Configuration]):
        """Creates the ``Simulation`` object only the configuration contents as either a
        full file path to the configuration file, a dictionary of the configuration
        contents, or pre-loaded ``Configuration`` object.

        Parameters
        ----------
        config : Union[str, dict, Configuration]
            The simulation configuration, see ``Configuration`` for more details on the
            contents. The following is a description of the acceptable contents:

             - ``str`` : the full file path of the configuration yaml file.
             - ``dict`` : a dictionary with the requried configuration settings.
             - ``Configuration`` : a pre-created ``Configuration`` object.

        Raises
        ------
        TypeError
            If ``config`` is not one of the three acceptable input types, then an error is
            raised.

        Returns
        -------
        Simulation
            A ready-to-run ``Simulation`` object.
        """
        if isinstance(config, (str, Path)):
            config = Path(config).resolve()  # type: ignore
            assert isinstance(config, Path)  # lets mypy know that I know what I'm doing
            config = load_yaml(config.parent, config.name)
        if isinstance(config, dict):
            config = Configuration.from_dict(config)
        if not isinstance(config, Configuration):
            raise TypeError(
                "``config`` must be a dictionary or ``Configuration`` object!"
            )
        assert isinstance(config, Configuration)  # lets mypy know it's a Configuration
        return cls(config.name, config.library, config)

    def _setup_simulation(self):
        """Initializes the simulation objects."""
        self.env = WombatEnvironment(
            self.config.library,
            self.config.weather,
            simulation_name=self.config.name,
            workday_start=self.config.workday_start,
            workday_end=self.config.workday_end,
            start_year=self.config.start_year,
            end_year=self.config.end_year,
        )
        self.repair_manager = RepairManager(self.env)
        self.windfarm = Windfarm(self.env, self.config.layout, self.repair_manager)
        self.service_equipment = []
        for service_equipment in self.config.service_equipment:
            self.service_equipment.append(
                ServiceEquipment(
                    self.env, self.windfarm, self.repair_manager, service_equipment
                )
            )

    def run(self, until: Optional[Union[int, float, Event]] = None):
        """Calls ``WombatEnvironment.run()`` and gathers the results for post-processing.
        See ``wombat.simulation.WombatEnvironment.run`` or ``simpy.Environment.run`` for more
        details.

        Parameters
        ----------
        until : Optional[Union[int, float, Event]], optional
            When to stop the simulation, by default None. See documentation on
            ``simpy.Environment.run`` for more details.
        """
        self.env.run(until=until)

        operations, events = self.env.convert_logs_to_csv(return_df=True)
        power_potential, power_production = self.env.power_production_potential_to_csv(
            windfarm=self.windfarm, operations=operations, return_df=True
        )
        self.metrics = Metrics(
            data_dir=self.config.library,
            events=events,
            operations=operations,
            potential=power_potential,
            production=power_production,
            inflation_rate=self.config.inflation_rate,
            project_capacity=self.config.project_capacity,
            turbine_capacities=[
                self.windfarm.node_system(t_id).capacity
                for t_id in self.windfarm.turbine_id
            ],
            fixed_costs=self.config.fixed_costs,
            substation_id=self.windfarm.substation_id.tolist(),
            turbine_id=self.windfarm.turbine_id.tolist(),
            service_equipment_names=[el.settings.name for el in self.service_equipment],
            SAM_settings=self.config.SAM_settings,
        )
