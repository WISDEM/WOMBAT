"""Creates the Turbine class."""
from __future__ import annotations

from typing import Callable  # type: ignore
from functools import reduce

import numpy as np
import pandas as pd

from wombat.core import RepairManager, WombatEnvironment
from wombat.utilities import IEC_power_curve
from wombat.windfarm.system import Subassembly


try:  # pylint: disable=duplicate-code
    from functools import cache  # type: ignore
except ImportError:  # pylint: disable=duplicate-code
    from functools import lru_cache  # pylint: disable=duplicate-code

    cache = lru_cache(None)  # pylint: disable=duplicate-code


@cache
def _product(x: float, y: float) -> float:
    """Multiplies two numbers. Used for a reduce operation.

    Parameters
    ----------
    x : float
        The first number.
    y : float
        The second number.

    Returns
    -------
    float
        The product of the two numbers.
    """
    return x * y


class System:
    """Can either be a turbine or substation, but is meant to be something that consists
    of 'Subassembly' pieces.

    See `here <https://www.sciencedirect.com/science/article/pii/S1364032117308985>`_
    for more information.
    """

    def __init__(
        self,
        env: WombatEnvironment,
        repair_manager: RepairManager,
        t_id: str,
        name: str,
        subassemblies: dict,
        system: str,
    ):
        """Initializes an individual windfarm asset.

        Parameters
        ----------
        env : WombatEnvironment
            The simulation environment.
        repair_manager : RepairManager
            The simulation repair and maintenance task manager.
        t_id : str
            The unique identifier for the asset.
        name : str
            The long form name/descriptor for the system/asset.
        subassemblies : dict
            The dictionary of subassemblies required for the system/asset.
        system : str
            The identifier should be one of "turbine" or "substation" to indicate the
            type of system this will be.

        Raises
        ------
        ValueError
            [description]
        """
        self.env = env
        self.repair_manager = repair_manager
        self.id = t_id
        self.name = name
        self.servicing = False
        self.cable_failure = False
        self.capacity = subassemblies["capacity_kw"]

        system = system.lower().strip()
        self._calculate_system_value(subassemblies)
        if system == "turbine":
            self._create_turbine_subassemblies(subassemblies)
            self._initialize_power_curve(subassemblies.get("power_curve", None))
        elif system == "substation":
            self._create_substation_subassemblies(subassemblies)
        else:
            raise ValueError("'system' must be one of 'turbine' or 'substation'!")

    def _calculate_system_value(self, subassemblies: dict) -> None:
        """Calculates the turbine's value based its capex_kw and capacity.

        Parameters
        ----------
        system : str
            One of "turbine" or "substation".
        subassemblies : dict
            Dictionary of subassemblies.
        """
        self.value = subassemblies["capacity_kw"] * subassemblies["capex_kw"]

    def _create_turbine_subassemblies(self, subassembly_data: dict) -> None:
        """Creates each subassembly as a separate attribute and also a list for quick
        access.

        Parameters
        ----------
        subassembly_data : dict
            Dictionary providing the maintenance and failure definitions for at least
            one of the following subassemblies:
             - electrical_system
             - electronic_control
             - sensors
             - hydraulic_system
             - yaw_system
             - rotor_blades
             - mechanical_brake
             - rotor_hub
             - gearbox
             - generator
             - supporting_structure
             - drive_train
        """
        subassembly_list = [
            "electrical_system",
            "electronic_control",
            "sensors",
            "hydraulic_system",
            "yaw_system",
            "rotor_blades",
            "mechanical_brake",
            "rotor_hub",
            "gearbox",
            "generator",
            "supporting_structure",
            "drive_train",
        ]

        # define the subassemblies  source: http://www.ecs.umass.edu/~arwade/wind_reliability.pdf
        self.subassemblies: list[Subassembly] = []

        # Check if the subassembly definition has been provided in the defintion
        # file, and if not, then skip it and move to the next subassembly
        try:
            self.electrical_system = Subassembly(
                self,
                self.env,
                "electrical_system",
                subassembly_data["electrical_system"],
            )
            self.subassemblies.append(self.electrical_system)
        except KeyError:
            pass

        try:
            self.electronic_control = Subassembly(
                self,
                self.env,
                "electronic_control",
                subassembly_data["electronic_control"],
            )
            self.subassemblies.append(self.electronic_control)
        except KeyError:
            pass

        try:
            self.sensors = Subassembly(
                self, self.env, "sensors", subassembly_data["sensors"]
            )
            self.subassemblies.append(self.sensors)
        except KeyError:
            pass

        try:
            self.hydraulic_system = Subassembly(
                self, self.env, "hydraulic_system", subassembly_data["hydraulic_system"]
            )
            self.subassemblies.append(self.hydraulic_system)
        except KeyError:
            pass

        try:
            self.yaw_system = Subassembly(
                self, self.env, "yaw_system", subassembly_data["yaw_system"]
            )
            self.subassemblies.append(self.yaw_system)
        except KeyError:
            pass

        try:
            self.rotor_blades = Subassembly(
                self, self.env, "rotor_blades", subassembly_data["rotor_blades"]
            )
            self.subassemblies.append(self.rotor_blades)
        except KeyError:
            pass

        try:
            self.mechanical_brake = Subassembly(
                self, self.env, "mechanical_brake", subassembly_data["mechanical_brake"]
            )
            self.subassemblies.append(self.mechanical_brake)
        except KeyError:
            pass

        try:
            self.rotor_hub = Subassembly(
                self, self.env, "rotor_hub", subassembly_data["rotor_hub"]
            )
            self.subassemblies.append(self.rotor_hub)
        except KeyError:
            pass

        try:
            self.gearbox = Subassembly(
                self, self.env, "gearbox", subassembly_data["gearbox"]
            )
            self.subassemblies.append(self.gearbox)
        except KeyError:
            pass

        try:
            self.generator = Subassembly(
                self, self.env, "generator", subassembly_data["generator"]
            )
            self.subassemblies.append(self.generator)
        except KeyError:
            pass

        try:
            self.supporting_structure = Subassembly(
                self,
                self.env,
                "supporting_structure",
                subassembly_data["supporting_structure"],
            )
            self.subassemblies.append(self.supporting_structure)
        except KeyError:
            pass

        try:
            self.drive_train = Subassembly(
                self, self.env, "drive_train", subassembly_data["drive_train"]
            )
            self.subassemblies.append(self.drive_train)
        except KeyError:
            pass

        if self.subassemblies == []:
            raise ValueError(
                "No subassembly data provided in the turbine configuration. At least"
                " defintion for the following subassemblies must be provided:"
                f" {subassembly_list}"
            )

        self.env.log_action(
            agent=self.name,
            action=f"subassemblies created: {[s.id for s in self.subassemblies]}",
            reason="windfarm initialization",
            system_id=self.id,
            system_name=self.name,
            system_ol=self.operating_level,
            part_ol=1,
            additional="initialization",
        )

    def _initialize_power_curve(self, power_curve_dict: dict) -> None:
        """Creates the power curve function based on the ``power_curve`` input in the
        ``subassembly_data`` dictionary. If there is no valid input, then 0 will always
        be reutrned.

        Parameters
        ----------
        power_curve_dict : dict
            The turbine definition dictionary.
        """
        self.power_curve: Callable
        if power_curve_dict is None:
            self.power_curve = IEC_power_curve(pd.Series([0]), pd.Series([0]))
        else:
            power_curve = power_curve_dict["file"]
            power_curve = self.env.data_dir / "windfarm" / power_curve
            power_curve = pd.read_csv(f"{power_curve}")
            power_curve = power_curve.loc[power_curve.power_kw != 0].reset_index(
                drop=True
            )
            bin_width = power_curve_dict.get("bin_width", 0.5)
            self.power_curve = IEC_power_curve(
                power_curve.windspeed_ms,
                power_curve.power_kw,
                windspeed_start=power_curve.windspeed_ms.min(),
                windspeed_end=power_curve.windspeed_ms.max(),
                bin_width=bin_width,
            )

    def _create_substation_subassemblies(self, subassembly_data: dict) -> None:
        """Creates each subassembly as a separate attribute and also a list for quick
        access.

        Parameters
        ----------
        subassembly_data : dict
            Dictionary containing the transformer maintenance and failure configurations.
        """
        self.subassemblies = []
        try:
            self.transformer = Subassembly(
                self, self.env, "transformer", subassembly_data["transformer"]
            )
            self.subassemblies = [self.transformer]
        except KeyError:
            pass

        if self.subassemblies == []:
            raise ValueError(
                "No subassembly data provided in the substation configuration; a"
                " transformer definition must be provided."
            )

        self.env.log_action(
            system_id=self.id,
            system_name=self.name,
            system_ol=self.operating_level,
            part_ol=1,
            agent=self.name,
            action="subassemblies created",
            reason="windfarm initialization",
            additional="initialization",
        )

    def interrupt_all_subassembly_processes(self) -> None:
        """Interrupts the running processes in all of the system's subassemblies."""
        [subassembly.interrupt_processes() for subassembly in self.subassemblies]  # type: ignore

    @property
    def operating_level(self) -> float:
        """The turbine's operating level, based on subassembly and cable performance.

        Returns
        -------
        float
            Operating level of the turbine.
        """
        if self.cable_failure or self.servicing:
            return 0.0
        else:
            return reduce(_product, [sub.operating_level for sub in self.subassemblies])

    @property
    def operating_level_wo_servicing(self) -> float:
        """The turbine's operating level, based on subassembly and cable performance,
        without accounting for servicing status.

        Returns
        -------
        float
            Operating level of the turbine.
        """
        if self.cable_failure:
            return 0.0
        else:
            return reduce(_product, [sub.operating_level for sub in self.subassemblies])

    def power(self, windspeed: list[float] | np.ndarray) -> np.ndarray:
        """Generates the power output for an iterable of windspeed values.

        Parameters
        ----------
        windspeed : list[float] | np.ndarrays
            Windspeed values, in m/s.

        Returns
        -------
        np.ndarray
            Power production, in kW.
        """
        return self.power_curve(windspeed)
