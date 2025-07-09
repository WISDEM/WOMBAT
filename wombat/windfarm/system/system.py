"""Creates the Turbine class."""

from __future__ import annotations

from typing import Callable
from operator import mul
from functools import reduce

import numpy as np
import pandas as pd

from wombat.core import SystemType, RepairManager, WombatEnvironment
from wombat.utilities import IEC_power_curve, calculate_hydrogen_production
from wombat.windfarm.system import Subassembly
from wombat.utilities.utilities import create_variable_from_string


class System:
    """Can either be a turbine, substation, or electrolyzer, but is meant to be
    something that consists of 'Subassembly' pieces.

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
            The identifier should be one of "turbine", "substation", or "electrolyzer"
            to indicate the type of system this will be.

        Raises
        ------
        ValueError
            [description]
        """
        self.env = env
        self.repair_manager = repair_manager
        self.id = t_id
        self.name = name
        self.system_type = SystemType(system)
        self.subassemblies: list[Subassembly] = []
        self.servicing = self.env.event()
        self.servicing_queue = self.env.event()
        self.cable_failure = self.env.event()

        if self.system_type is SystemType.ELECTROLYZER:
            self.n_stacks = subassemblies["n_stacks"]
            self.capacity = self.n_stacks * subassemblies["stack_capacity_kw"]
        else:
            self.capacity = subassemblies["capacity_kw"]

        power_curve = subassemblies.get("power_curve")
        if self.system_type is SystemType.TURBINE:
            self._initialize_power_curve(power_curve)

        if self.system_type is SystemType.ELECTROLYZER:
            self._initialize_production_curve(power_curve)

        self.value = self._calculate_system_value(subassemblies.get("capex_kw"))

        # Ensure servicing statuses starts as processed and inactive
        self.servicing.succeed()
        self.servicing_queue.succeed()
        self.cable_failure.succeed()

        self._create_subassemblies(subassemblies)

    def _calculate_system_value(self, capex_kw: int | float | None) -> int | float:
        """Calculates the system's value based its :py:attr:`capex_kw` and either
        :py:attr:`capacity`. For turbines this is the ``capacity_kw`, and for
        electrolyzers, this is the ``rated_production`` based on the
        `stack_capacit_kw``, ``n_stacks``, and production curve.

        Parameters
        ----------
        capex_kw : int | float
            The CapEx per kw of the system.

        Returns
        -------
        int | float
            The total system CapEx.

        Raises
        ------
        TypeError
            Raised if :py:attr:`capex_kw` is None.
        """
        if capex_kw is None or not isinstance(capex_kw, (float, int)):
            msg = f"Invalid `capex_kw` provided for {self.system_type}: {self.name}"
            raise TypeError(msg)
        return self.capacity * capex_kw

    def _create_subassemblies(self, subassembly_data: dict) -> None:
        """Creates each subassembly as a separate attribute and also a list for quick
        access.

        Parameters
        ----------
        subassembly_data : dict
            Dictionary providing the maintenance and failure definitions for at least
            one subassembly named
        """
        # Set the subassembly data variables from the remainder of the keys in the
        # system configuration file/dictionary
        exclude_keys = [
            "capacity_kw",
            "capex_kw",
            "power_curve",
            "n_stacks",
            "stack_capacity_kw",
        ]
        invalid = (
            "env",
            "repair_manager",
            "id",
            "name",
            "capacity",
            "n_stacks",
            "subassemblies",
            "servicing",
            "servicing_queue",
            "cable_failure",
            "value",
            "system_type",
            "power",
            "power_curve",
            "rated_production",
        )
        for key, data in subassembly_data.items():
            if key in exclude_keys:
                continue
            if key.strip().lower() in invalid:
                raise ValueError(f"Input subassembly `id` cannot be used: {key}.")
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

    def _initialize_power_curve(self, power_curve_dict: dict | None) -> None:
        """Creates the power curve function based on the :py:attr:power_curve_dict`
        input in the ``subassembly_data`` dictionary. If there is no valid input, then 0
        will always be reutrned.

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

    def _initialize_production_curve(self, power_curve_dict: dict | None) -> None:
        """Creates the H2 production curve function based on the
        :py:attr:power_curve_dict` input to the  ``subassembly_data`` dictionary. If no
        input, then a 0 will always be returned. Sets ``rated_production`` to kg/hr.
        """
        if power_curve_dict is None:
            power_curve_dict = {}
        p1 = power_curve_dict.get("p1")
        p2 = power_curve_dict.get("p2")
        p3 = power_curve_dict.get("p3")
        p4 = power_curve_dict.get("p4")
        p5 = power_curve_dict.get("p5")
        efficiency_rate = power_curve_dict.get("efficiency_rate")
        fe = power_curve_dict.get("FE", 0.9999999)
        n_cells = power_curve_dict.get("n_cells", 135)
        turndown_ratio = power_curve_dict.get("turndown_ratio", 0.1)
        rated_capacity = self.capacity
        self.power_curve = calculate_hydrogen_production(
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            p5=p5,
            efficiency_rate=efficiency_rate,
            rated_capacity=rated_capacity,
            FE=fe,
            n_cells=n_cells,
            turndown_ratio=turndown_ratio,
        )
        self.rated_production, *_ = self.power_curve(np.array([rated_capacity]))

    def interrupt_all_subassembly_processes(
        self,
        origin: Subassembly | None = None,
        subassembly_full_reset: list[str] | None = None,
    ) -> None:
        """Interrupts the running processes in all of the system's subassemblies.

        Parameters
        ----------
        origin : Subassembly
            The subassembly that triggered the request, if the method call is coming
            from a subassembly shutdown event.
        subassembly_full_reset: list[str] | None
            List of all subassebly `id` that will get a full reset of their simulated
            failure and maintenance processes by a replacement or tow-to-port event.
        """
        [
            subassembly.interrupt_processes(
                origin=origin, subassembly_full_reset=subassembly_full_reset
            )  # type: ignore
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
