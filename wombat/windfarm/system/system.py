"""Creates the Turbine class."""

from __future__ import annotations

from typing import Callable
from operator import mul
from functools import reduce

import numpy as np
import pandas as pd

from wombat.core import RepairManager, WombatEnvironment
from wombat.utilities import IEC_power_curve
from wombat.windfarm.system import Subassembly
from wombat.utilities.utilities import create_variable_from_string


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
        self.capacity = subassemblies["capacity_kw"]
        self.subassemblies: list[Subassembly] = []
        self.servicing = self.env.event()
        self.servicing_queue = self.env.event()
        self.cable_failure = self.env.event()

        # Ensure servicing statuses starts as processed and inactive
        self.servicing.succeed()
        self.servicing_queue.succeed()
        self.cable_failure.succeed()

        system = system.lower().strip()
        self._calculate_system_value(subassemblies)
        if system not in ("turbine", "substation"):
            raise ValueError("'system' must be one of 'turbine' or 'substation'!")

        self._create_subassemblies(subassemblies, system)

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

    def _create_subassemblies(self, subassembly_data: dict, system: str) -> None:
        """Creates each subassembly as a separate attribute and also a list for quick
        access.

        Parameters
        ----------
        subassembly_data : dict
            Dictionary providing the maintenance and failure definitions for at least
            one subassembly named
        system : str
            One of "turbine" or "substation" to indicate if the power curves should also
            be created, or not.
        """
        # Set the subassembly data variables from the remainder of the keys in the
        # system configuration file/dictionary
        exclude_keys = ["capacity_kw", "capex_kw", "power_curve"]
        for key, data in subassembly_data.items():
            if key in exclude_keys:
                continue
            name = create_variable_from_string(key)
            subassembly = Subassembly(self, self.env, name, data)
            setattr(self, name, subassembly)
            self.subassemblies.append(getattr(self, name))

        if self.subassemblies == []:
            raise ValueError(
                "At least one subassembly definition requred for ",
                f"ID: {self.id}, Name: {self.name}.",
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

        # If the system is a turbine, create the power curve, if available
        if system == "turbine":
            self._initialize_power_curve(subassembly_data.get("power_curve", None))

    def _initialize_power_curve(self, power_curve_dict: dict | None) -> None:
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
            power_curve_file = self.env.data_dir / "turbines" / power_curve_dict["file"]
            power_curve = pd.read_csv(power_curve_file)
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

    def interrupt_all_subassembly_processes(
        self, origin: Subassembly | None = None, replacement: str | None = None
    ) -> None:
        """Interrupts the running processes in all of the system's subassemblies.

        Parameters
        ----------
        origin : Subassembly
            The subassembly that triggered the request, if the method call is coming
            from a subassembly shutdown event.
        replacement: bool, optional
            If a subassebly `id` is provided, this indicates the interruption is caused
            by its replacement event. Defaults to None.
        """
        [
            subassembly.interrupt_processes(origin=origin, replacement=replacement)  # type: ignore
            for subassembly in self.subassemblies
        ]

    @property
    def operating_level(self) -> float:
        """The turbine's operating level, based on subassembly and cable performance.

        Returns
        -------
        float
            Operating level of the turbine.
        """
        if self.cable_failure.triggered and self.servicing.triggered:
            ol: float = reduce(mul, [sub.operating_level for sub in self.subassemblies])
            return ol  # type: ignore
        return 0.0

    @property
    def operating_level_wo_servicing(self) -> float:
        """The turbine's operating level, based on subassembly and cable performance,
        without accounting for servicing status.

        Returns
        -------
        float
            Operating level of the turbine.
        """
        if self.cable_failure.triggered:
            ol: float = reduce(mul, [sub.operating_level for sub in self.subassemblies])
            return ol  # type: ignore
        return 0.0

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
