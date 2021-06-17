"""The main API for the `wombat`."""

from pathlib import Path
from typing import List, Optional, Union

import attr
import pandas as pd
from simpy.events import Event  # type: ignore

from wombat.core import Metrics, RepairManager, ServiceEquipment, WombatEnvironment
from wombat.core.library import library_map, load_yaml
from wombat.windfarm import Windfarm


def _library_mapper(file_path: Union[str, Path]) -> Union[str, Path]:
    """Attempts to extract a default library path if one of "DINWOODIE" or "IEA_26"
    are passed, other returns `file_path`.

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
    return library_map.get(file_path, file_path)


@attr.s(frozen=True, auto_attribs=True)
class Configuration:
    """The `Simulation` configuration data class that provides all the necessary definitions.

    Parameters
    ----------
    name: str
        Name of the simulation. Used for logging files.
    library : str
        The data directory. See `wombat.simulation.WombatEnvironment` for more details.
    layout : str
        The windfarm layout file. See `wombat.Windfarm` for more details.
    service_equipment : Union[str, List[str]]
        The equpiment that will be used in the simulation. See
        `wombat.core.ServiceEquipment` for more details.
    weather : str
        The weather profile to be used. See `wombat.simulation.WombatEnvironment`
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
        the first valid date of this year in `weather`.
    end_year : int
        Final year of the simulation. The exact date will be determined by
        the last valid date of this year in `weather`.
    SAM_settings : str
        The SAM settings file to be used for financial modeling, optional, by
        default None.
    """

    name: str
    library: Path
    layout: str = attr.ib(converter=_library_mapper)
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


class Simulation:
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

    def __init__(self, name: str, library_path: str, config: Union[str, dict]):
        """Creates a `Simulation` object with an analysis ready to run.

        Parameters
        ----------
        name: str
            Name of the simulation. Used for logging files.
        library_path : str
            The path to the main data library. If one of DINWOODIE or IEA_26
        config : Union[str, dict]
            The path to a configuration dictionary or the dictionary itself.

        Returns
        -------
        Simulation
            A `Simulation` object
        """
        library_path = _library_mapper(library_path)
        self.data_dir = Path(library_path).resolve()
        if isinstance(config, str):
            config = load_yaml(self.data_dir / "config", config)  # type: ignore
        config["name"] = name  # type: ignore
        config["library"] = self.data_dir  # type: ignore
        self.config = Configuration(**config)  # type: ignore

        self.setup_simulation()

    @classmethod
    def from_inputs(
        cls,
        name: str,
        library: str,
        layout: str,
        service_equipment: Union[str, List[str]],
        weather: str,
        start_year: int,
        end_year: int,
        workday_start: int,
        workday_end: int,
        inflation_rate: float,
        fixed_costs: str,
        project_capacity: Union[int, float],
        SAM_settings: str = None,
    ):
        """Creates a `Simulation` object from a series of inputs instead of a
        predefined dictionary or YAML file.

        Parameters
        ----------
        name: str
            Name of the simulation. Used for logging files.
        library : str
            The data directory. See `wombat.simulation.WombatEnvironment` for more details.
        layout : str
            The windfarm layout file. See `wombat.Windfarm` for more details.
        service_equipment : Union[str, List[str]]
            The equpiment that will be used in the simulation. See
            `wombat.core.ServiceEquipment` for more details.
        weather : str
            The weather profile to be used. See `wombat.simulation.WombatEnvironment`
            for more details.
        start_year : int
            Start year of the simulation. The exact date will be determined by
            the first valid date of this year in `weather`.
        end_year : int
            Final year of the simulation. The exact date will be determined by
            the last valid date of this year in `weather`.
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
        SAM_settings : str
            The SAM settings file to be used for financial modeling, optional, by
            default None.

        Returns
        -------
        Simulation
            Returns a `Simulation` object.
        """
        config = dict(
            name=name,
            library=_library_mapper(library),
            layout=layout,
            service_equipment=service_equipment,
            weather=weather,
            start_year=start_year,
            end_year=end_year,
            workday_start=workday_start,
            workday_end=workday_end,
            inflation_rate=inflation_rate,
            fixed_costs=fixed_costs,
            project_capacity=project_capacity,
            SAM_settings=SAM_settings,
        )
        return cls(name, library, config)

    def setup_simulation(self):
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
        """Calls `WombatEnvironment.run()` and gathers the results for post-processing.
        See `wombat.simulation.WombatEnvironment.run` or `simpy.Environment.run` for more
        details.

        Parameters
        ----------
        until : Optional[Union[int, float, Event]], optional
            When to stop the simulation, by default None. See documentation on
            `simpy.Environment.run` for more details.
        """
        self.env.run(until=until)

        operations, events = self.env.convert_logs_to_csv(return_df=True)
        power_potential, power_production = self.env.power_production_potential_to_csv(
            windfarm=self.windfarm, operations=operations, return_df=True
        )
        self.metrics = Metrics(
            self.config.library,
            events,
            operations,
            power_potential,
            power_production,
            self.config.inflation_rate,
            self.config.project_capacity,
            [
                self.windfarm.node_system(t_id).capacity
                for t_id in self.windfarm.turbine_id
            ],
            self.config.fixed_costs,
            self.windfarm.substation_id.tolist(),
            self.windfarm.turbine_id.tolist(),
            [el.settings.name for el in self.service_equipment],
            SAM_settings=self.config.SAM_settings,
        )
