"""The main API for the ``wombat``."""
from __future__ import annotations

from typing import Optional
from pathlib import Path

import yaml
import pandas as pd
from attrs import Attribute, field, define
from simpy.events import Event  # type: ignore

from wombat.core import (
    Metrics,
    FromDictMixin,
    RepairManager,
    ServiceEquipment,
    WombatEnvironment,
)
from wombat.windfarm import Windfarm
from wombat.core.port import Port
from wombat.core.library import load_yaml, library_map


def _library_mapper(file_path: str | Path) -> Path:
    """Attempts to extract a default library path if one of "DINWOODIE" or "IEA_26"
    are passed, other returns ``file_path``.

    Parameters
    ----------
    file_path : str | Path
        Should be a valid file path, or one of "DINWOODIE" or "IEA_26" to indicate a
        provided library is being used.

    Returns
    -------
    str | Path
        The library path.
    """
    return Path(library_map.get(file_path, file_path)).resolve()  # type: ignore


@define(frozen=True, auto_attribs=True)
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
    service_equipment : str | list[str]
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
    project_capacity : int | float
        The total capacity of the wind plant, in MW.
    port : dict | str | Path
        The port configuration file or dictionary that will be used to setup a
        tow-to-port repair strategy, default None.
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
    library: Path = field(converter=_library_mapper)
    layout: str
    service_equipment: str | list[str]
    weather: str | pd.DataFrame
    workday_start: int
    workday_end: int
    inflation_rate: float
    fixed_costs: str
    project_capacity: int | float
    port: dict | str | Path = field(default=None)
    start_year: int = field(default=None)
    end_year: int = field(default=None)
    SAM_settings: str = field(default=None)

    def __attrs_post_init__(self):
        if isinstance(self.service_equipment, str):
            object.__setattr__(self, "service_equipment", [self.service_equipment])


@define(auto_attribs=True)
class Simulation(FromDictMixin):
    """The primary API to interact with the simulation methodologies.

    Parameters
    ----------
    library_path : str
        The path to the main data library.
    config : Configuration | dict | str
        One of the following:
         - A pre-loaded ``Configuration`` object
         - A dictionary ready to be converted to a ``Configuration`` object
         - The name of the configuration file to be loaded, that will be located at:
           ``library_path`` / config / ``config``
    """

    library_path: Path = field(converter=_library_mapper)
    config: Configuration = field()

    metrics: Metrics = field(init=False)
    windfarm: Windfarm = field(init=False)
    env: WombatEnvironment = field(init=False)
    repair_manager: RepairManager = field(init=False)
    service_equipment: list[ServiceEquipment] = field(init=False)
    port: Port = field(init=False)

    def __attrs_post_init__(self) -> None:
        self._setup_simulation()

    @config.validator  # type: ignore
    def _create_configuration(
        self, attribute: Attribute, value: str | dict | Configuration
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

        if self.config.library != self.library_path:
            raise ValueError(
                "``library_path`` and the library in ``config`` do not match!"
            )

    @classmethod
    def from_config(cls, config: str | dict | Configuration):
        """Creates the ``Simulation`` object only the configuration contents as either a
        full file path to the configuration file, a dictionary of the configuration
        contents, or pre-loaded ``Configuration`` object.

        Parameters
        ----------
        config : str | dict | Configuration
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
        # NOTE: mypy is not caught up with attrs yet :(
        return cls(config.library, config)  # type: ignore

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
        if self.config.port is not None:
            self.port = Port(
                self.env, self.windfarm, self.repair_manager, self.config.port
            )
            self.service_equipment.extend(self.port.tugboat_manager.items)

    def run(
        self,
        until: Optional[int | float | Event] = None,
        create_metrics: bool = True,
        save_metrics_inputs: bool = True,
    ):
        """Calls ``WombatEnvironment.run()`` and gathers the results for post-processing.
        See ``wombat.simulation.WombatEnvironment.run`` or ``simpy.Environment.run`` for more
        details.

        Parameters
        ----------
        until : Optional[int | float | Event], optional
            When to stop the simulation, by default None. See documentation on
            ``simpy.Environment.run`` for more details.
        create_metrics : bool, optional
            If True, the metrics object will be created, and not, if False, by default True.
        save_metrics_inputs : bool, optional
            If True, the metrics inputs data will be saved to a yaml file, with file
            references to any larger data structures that can be reloaded later. If
            False, the data will not be saved, by default True.
        """
        self.env.run(until=until)
        if save_metrics_inputs:
            self.save_metrics_inputs()
        if create_metrics:
            self.initialize_metrics()

    def initialize_metrics(self) -> None:
        """Instantiates the ``metrics`` attribute after the simulation is run."""
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
                self.windfarm.system(t_id).capacity for t_id in self.windfarm.turbine_id
            ],
            fixed_costs=self.config.fixed_costs,
            substation_id=self.windfarm.substation_id.tolist(),
            turbine_id=self.windfarm.turbine_id.tolist(),
            service_equipment_names=[el.settings.name for el in self.service_equipment],  # type: ignore
            SAM_settings=self.config.SAM_settings,
        )

    def save_metrics_inputs(self) -> None:
        """Saves the inputs for the `Metrics` initialization with either a direct
        copy of the data or a file reference that can be loaded later.
        """
        data = dict(
            data_dir=str(self.config.library),
            events=str(self.env.events_log_fname.with_suffix(".csv")),
            operations=str(self.env.operations_log_fname.with_suffix(".csv")),
            potential=str(self.env.power_potential_fname),
            production=str(self.env.power_production_fname),
            inflation_rate=self.config.inflation_rate,
            project_capacity=self.config.project_capacity,
            turbine_capacities=[
                self.windfarm.system(t_id).capacity for t_id in self.windfarm.turbine_id
            ],
            fixed_costs=self.config.fixed_costs,
            substation_id=self.windfarm.substation_id.tolist(),
            turbine_id=self.windfarm.turbine_id.tolist(),
            service_equipment_names=[el.settings.name for el in self.service_equipment],  # type: ignore
            SAM_settings=self.config.SAM_settings,
        )

        with open(self.env.metrics_input_fname, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
