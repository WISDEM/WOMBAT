"""The main API for the ``wombat``."""

from __future__ import annotations

import datetime
from copy import deepcopy
from typing import TYPE_CHECKING
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from attrs import Attribute, field, define, validators
from simpy.events import Event

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
from wombat.core.data_classes import convert_to_list


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
    """The ``Simulation`` configuration data class that provides all the necessary
    definitions.

    Parameters
    ----------
    name: str
        Name of the simulation. Used for logging files.
    layout : str
        The windfarm layout file. See ``wombat.Windfarm`` for more details.
    service_equipment : str | list[str | list[str, int]]
        The equpiment that will be used in the simulation. For multiple instances of a
        single vessel use a list of the file name/id and the number of vessels. See
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
    port_distance : int | float
        The simulation-wide daily travel distance for servicing equipment. This should
        be used as a base setting when multiple or all servicing equipment will be
        operating out of the same base location, but can be individually modified.
    start_year : int
        Start year of the simulation. The exact date will be determined by
        the first valid date of this year in ``weather``.
    end_year : int
        Final year of the simulation. The exact date will be determined by
        the last valid date of this year in ``weather``.
    non_operational_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the environment level, an
        undefined or later starting date will be overridden for all servicing equipment
        and any modeled port, by default None.
    non_operational_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the environment level, an
        undefined or earlier ending date will be overridden for all servicing equipment
        and any modeled port, by default None.
    reduced_speed_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the environment level, an
        undefined or later starting date will be overridden for all servicing equipment
        and any modeled port, by default None.
    reduced_speed_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the environment level, an
        undefined or earlier ending date will be overridden for all servicing equipment
        and any modeled port, by default None.
    reduced_speed : float
        The maximum operating speed during the annualized reduced speed operations.
        When defined at the environment level, an undefined or faster value will be
        overridden for all servicing equipment and any modeled port, by default 0.0.
    random_seed : int | None
        The random seed to be passed to a universal NumPy ``default_rng`` object to
        generate Weibull random generators, by default None.
    random_generator : np.random._generator.Generator | None
        An optional numpy random generator that can be provided to seed a simulation
        with the same generator each time, in place of the random seed. If a
        :py:attr:`random_seed` is also provided, this will override the random seed,
        by default None.
    cables : dict[str, dict] | None
        A dictionary of cable configurations with the keys aligning with the layout
        file's :py:attr:`upstream_cable` field, which would replace the need for YAML
        file defintions, by default None.
    substations : dict[str, dict] | None
        A dictionary of substation configurations with the keys aligning with the layout
        file's :py:attr:`upstream_cable` field, which would replace the need for YAML
        file defintions, by default None.
    turbines : dict[str, dict] | None
        A dictionary of turbine configurations with the keys aligning with the layout
        file's :py:attr:`upstream_cable` field, which would replace the need for YAML
        file defintions, by default None.
    vessels : dict[str, dict] | None
        A dictionary of servicing equipment configurations with the keys aligning with
        entries of :py:attr:`service_equipment` field, which would replace the need for
        YAML file defintions, by default None.
    """

    name: str
    layout: str
    service_equipment: str | list[str] = field(converter=convert_to_list)
    weather: str | pd.DataFrame
    workday_start: int = field(converter=int)
    workday_end: int = field(converter=int)
    inflation_rate: float = field(converter=float)
    project_capacity: int | float = field(converter=float)
    fixed_costs: dict | str | Path = field(default=None)
    port: dict | str | Path = field(default=None)
    start_year: int = field(default=None)
    end_year: int = field(default=None)
    port_distance: int | float = field(default=None)
    non_operational_start: str | datetime.datetime | None = field(default=None)
    non_operational_end: str | datetime.datetime | None = field(default=None)
    reduced_speed_start: str | datetime.datetime | None = field(default=None)
    reduced_speed_end: str | datetime.datetime | None = field(default=None)
    reduced_speed: float = field(default=0.0)
    random_seed: int | None = field(default=None)
    random_generator: np.random._generator.Generator | None = field(default=None)
    cables: dict[str, dict] | None = field(
        default=None, validator=validators.instance_of((dict, type(None)))
    )
    substations: dict[str, dict] | None = field(
        default=None, validator=validators.instance_of((dict, type(None)))
    )
    turbines: dict[str, dict] | None = field(
        default=None, validator=validators.instance_of((dict, type(None)))
    )
    vessels: dict[str, dict] | None = field(
        default=None, validator=validators.instance_of((dict, type(None)))
    )


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
    random_seed : int | None
        The random seed to be passed to a universal NumPy ``default_rng`` object to
        generate Weibull random generators, by default None.
    random_generator: np.random._generator.Generator | None
        An optional numpy random generator that can be provided to seed a simulation
        with the same generator each time, in place of the random seed. If a
        :py:attr:`random_seed` is also provided, this will override the random seed,
        by default None.
    """

    library_path: Path = field(converter=_library_mapper)
    config: Configuration = field()
    random_seed: int | None = field(default=None)
    random_generator: np.random._generator.Generator | None = field(default=None)

    metrics: Metrics = field(init=False)
    windfarm: Windfarm = field(init=False)
    env: WombatEnvironment = field(init=False)
    repair_manager: RepairManager = field(init=False)
    service_equipment: list[ServiceEquipment] = field(init=False)
    port: Port = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Post-initialization hook."""
        # Check for random seeding from the configuration if none provided directly
        if self.random_seed is None:
            self.random_seed = self.config.random_seed
        if self.random_generator is None:
            self.random_generator = self.config.random_generator

        # Finish the setup
        self._setup_simulation()

    @config.validator  # type: ignore
    def _create_configuration(
        self, attribute: Attribute, value: str | Path | dict | Configuration
    ) -> None:
        """Validates the configuration object and creates the ``Configuration`` object
        for the simulation.

        Raises
        ------
        TypeError
            Raised if the value provided is not able to create a valid ``Configuration``
            object
        ValueError
            Raised if ``name`` and ``config.name`` or ``library_path`` and
            ``config.library`` are not aligned.

        Returns
        -------
        Configuration
            The validated simulation configuration
        """
        if isinstance(value, (str, Path)):
            value = load_yaml(self.library_path / "project/config", value)
        if isinstance(value, dict):
            value = Configuration.from_dict(value)
        if isinstance(value, Configuration):
            object.__setattr__(self, attribute.name, value)
        else:
            raise TypeError(
                "``config`` must be a dictionary, valid file path to a yaml-enocoded",
                "dictionary, or ``Configuration`` object!",
            )

    @classmethod
    def from_config(
        cls, library_path: str | Path, config: str | Path | dict | Configuration
    ):
        """Creates the ``Simulation`` object only the configuration contents as either a
        full file path to the configuration file, a dictionary of the configuration
        contents, or pre-loaded ``Configuration`` object.

        Parameters
        ----------
        library_path : str | Path
            The simulation's data library. If a filename is provided for
            :py:attr:`config`, this is the data library from where it will be imported.
            This will also be used to feed into the returned `Simulation.library_path`.
        config : str | Path | dict | Configuration
            The simulation configuration, see ``Configuration`` for more details on the
            contents. The following is a description of the acceptable contents:

            - ``str`` : the full file path of the configuration yaml file.
            - ``dict`` : a dictionary with the requried configuration settings.
            - ``Configuration`` : a pre-created ``Configuration`` object.

        Raises
        ------
        TypeError
            Raised if ``config`` is not one of the three acceptable input types.

        Returns
        -------
        Simulation
            A ready-to-run ``Simulation`` object.
        """
        library_path = _library_mapper(library_path)
        if isinstance(config, (str, Path)):
            config = library_path / "project" / "config" / config
            config = load_yaml(config.parent, config.name)
        if isinstance(config, dict):
            config = Configuration.from_dict(config)
        if not isinstance(config, Configuration):
            raise TypeError(
                "``config`` must be a dictionary or ``Configuration`` object!"
            )
        if TYPE_CHECKING:
            assert isinstance(config, Configuration)  # mypy helper
        return cls(  # type: ignore
            library_path=library_path,
            config=config,
            random_seed=config.random_seed,
            random_generator=config.random_generator,
        )

    def _initialize_servicing_equipment(self, configuration: str | dict) -> None:
        """
        Initializes a single piece of servicing equipment.

        Parameters
        ----------
        configuration : str | dict
            The servicing equipment configuration dictionary or YAML file.
        """
        equipment = ServiceEquipment(
            self.env, self.windfarm, self.repair_manager, configuration
        )
        equipment.finish_setup_with_environment_variables()
        name = equipment.settings.name
        if name in self.service_equipment:
            msg = (
                f"Servicing equipment `{name}` already exists, please use"
                " unique names for all servicing equipment."
            )
            raise ValueError(msg)
        self.service_equipment[name] = equipment  # type: ignore

    def _setup_servicing_equipment(self):
        """Initializes the servicing equipment used in the simulation."""
        for service_equipment in self.config.service_equipment:
            n = 1
            if isinstance(service_equipment, list):
                n, service_equipment = service_equipment
                if isinstance(n, str) and isinstance(service_equipment, int):
                    n, service_equipment = service_equipment, n

            if not service_equipment.endswith((".yml", ".yaml")):
                if self.config.vessels is None:
                    msg = (
                        "The input to `vessels` must be defined if file names not"
                        f" provided to `servicing_equipment`. '{service_equipment}'"
                        "is not a YAML filename."
                    )
                    raise ValueError(msg)
                service_equipment = self.config.vessels[service_equipment]  # type: ignore

            if n == 1:
                self._initialize_servicing_equipment(service_equipment)
                continue

            # YAML files must be loaded for repeats so that the naming can be unique
            if isinstance(service_equipment, str):
                if service_equipment.endswith((".yaml", ".yml")):
                    service_equipment = load_yaml(
                        self.env.data_dir / "vessels", service_equipment
                    )
            if TYPE_CHECKING:
                assert isinstance(service_equipment, dict)
            for i in range(n):
                config = deepcopy(service_equipment)
                name = f"{config['name']} {i + 1}"
                config["name"] = name
                self._initialize_servicing_equipment(config)

    def _setup_simulation(self):
        """Initializes the simulation objects."""
        self.env = WombatEnvironment(
            self.library_path,
            self.config.weather,
            simulation_name=self.config.name,
            workday_start=self.config.workday_start,
            workday_end=self.config.workday_end,
            start_year=self.config.start_year,
            end_year=self.config.end_year,
            port_distance=self.config.port_distance,
            non_operational_start=self.config.non_operational_start,
            non_operational_end=self.config.non_operational_end,
            reduced_speed_start=self.config.reduced_speed_start,
            reduced_speed_end=self.config.reduced_speed_end,
            reduced_speed=self.config.reduced_speed,
            random_seed=self.random_seed,
            random_generator=self.random_generator,
        )
        self.repair_manager = RepairManager(self.env)
        self.windfarm = Windfarm(
            env=self.env,
            windfarm_layout=self.config.layout,
            repair_manager=self.repair_manager,
            substations=self.config.substations,
            turbines=self.config.turbines,
            cables=self.config.cables,
        )
        self.service_equipment: dict[str, ServiceEquipment] = {}  # type: ignore
        self._setup_servicing_equipment()

        # Create the port and add any tugboats to the available servicing equipment list
        if self.config.port is not None:
            self.port = Port(
                self.env, self.windfarm, self.repair_manager, self.config.port
            )
            for service_equipment in self.port.service_equipment_manager.items:
                name = service_equipment.settings.name  # type: ignore
                if name in self.service_equipment:
                    raise ValueError(
                        f"Servicing equipment `{name}` already exists, please use"
                        " unique names for all servicing equipment."
                    )
                self.service_equipment[name] = service_equipment  # type: ignore

        if self.config.project_capacity * 1000 != round(self.windfarm.capacity, 6):
            raise ValueError(
                f"Input `project_capacity`: {self.config.project_capacity:,.6f} MW is"
                f" not equal to the sum of turbine capacities:"
                f" {self.windfarm.capacity / 1000:,.6f} MW"
            )

    def run(
        self,
        until: int | float | Event | None = None,
        create_metrics: bool = True,
        save_metrics_inputs: bool = True,
    ):
        """Calls ``WombatEnvironment.run()`` and gathers the results for
        post-processing. See ``wombat.simulation.WombatEnvironment.run`` or
        ``simpy.Environment.run`` for more details.

        Parameters
        ----------
        until : Optional[int | float | Event], optional
            When to stop the simulation, by default None. See documentation on
            ``simpy.Environment.run`` for more details.
        create_metrics : bool, optional
            If True, the metrics object will be created, and not, if False, by default
            True.
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
        events = self.env.load_events_log_dataframe()
        operations = self.env.load_operations_log_dataframe()
        power_potential, power_production = self.env.power_production_potential_to_csv(
            windfarm=self.windfarm, operations=operations, return_df=True
        )
        substation_turbine_map = {
            s_id: {k: v.tolist() for k, v in dict.items()}
            for s_id, dict in self.windfarm.substation_turbine_map.items()
        }
        capacities = [
            self.windfarm.system(t).capacity for t in self.windfarm.turbine_id
        ]
        self.metrics = Metrics(
            data_dir=self.library_path,
            events=events,
            operations=operations,
            potential=power_potential,
            production=power_production,
            inflation_rate=self.config.inflation_rate,
            project_capacity=self.config.project_capacity,
            turbine_capacities=capacities,
            fixed_costs=self.config.fixed_costs,  # type: ignore
            substation_id=self.windfarm.substation_id.tolist(),
            turbine_id=self.windfarm.turbine_id.tolist(),
            substation_turbine_map=substation_turbine_map,
            service_equipment_names=[*self.service_equipment],  # type: ignore
        )

    def save_metrics_inputs(self) -> None:
        """Saves the inputs for the `Metrics` initialization with either a direct
        copy of the data or a file reference that can be loaded later.
        """
        substation_turbine_map = {
            s_id: {k: v.tolist() for k, v in dict.items()}
            for s_id, dict in self.windfarm.substation_turbine_map.items()
        }
        data = {
            "data_dir": str(self.library_path),
            "events": str(self.env.events_log_fname),
            "operations": str(self.env.operations_log_fname),
            "potential": str(self.env.power_potential_fname),
            "production": str(self.env.power_production_fname),
            "inflation_rate": self.config.inflation_rate,
            "project_capacity": self.config.project_capacity,
            "turbine_capacities": [
                self.windfarm.system(t_id).capacity for t_id in self.windfarm.turbine_id
            ],
            "fixed_costs": self.config.fixed_costs,
            "substation_id": self.windfarm.substation_id.tolist(),
            "turbine_id": self.windfarm.turbine_id.tolist(),
            "substation_turbine_map": substation_turbine_map,
            "service_equipment_names": [*self.service_equipment],
        }

        with open(self.env.metrics_input_fname, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
