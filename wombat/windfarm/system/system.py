"""Creates the Turbine class."""


import numpy as np
import pandas as pd
from typing import List, Union, Callable  # type: ignore
from functools import reduce

from wombat.core import RepairManager, WombatEnvironment
from wombat.utilities import IEC_power_curve
from wombat.windfarm.system import Subassembly


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
            Dictionary
        """
        # define the subassemblies  source: http://www.ecs.umass.edu/~arwade/wind_reliability.pdf
        self.electrical_system = Subassembly(
            self, self.env, "electrical_system", subassembly_data["electrical_system"]
        )
        self.electronic_control = Subassembly(
            self, self.env, "electronic_control", subassembly_data["electronic_control"]
        )
        self.sensors = Subassembly(
            self, self.env, "sensors", subassembly_data["sensors"]
        )
        self.hydraulic_system = Subassembly(
            self, self.env, "hydraulic_system", subassembly_data["hydraulic_system"]
        )
        self.yaw_system = Subassembly(
            self, self.env, "yaw_system", subassembly_data["yaw_system"]
        )
        self.rotor_blades = Subassembly(
            self, self.env, "rotor_blades", subassembly_data["rotor_blades"]
        )
        self.mechanical_brake = Subassembly(
            self, self.env, "mechanical_brake", subassembly_data["mechanical_brake"]
        )
        self.rotor_hub = Subassembly(
            self, self.env, "rotor_hub", subassembly_data["rotor_hub"]
        )
        self.gearbox = Subassembly(
            self, self.env, "gearbox", subassembly_data["gearbox"]
        )
        self.generator = Subassembly(
            self, self.env, "generator", subassembly_data["generator"]
        )
        self.supporting_structure = Subassembly(
            self,
            self.env,
            "supporting_structure",
            subassembly_data["supporting_structure"],
        )
        self.drive_train = Subassembly(
            self, self.env, "drive_train", subassembly_data["drive_train"]
        )

        self.subassemblies = [
            self.electrical_system,
            self.electronic_control,
            self.sensors,
            self.hydraulic_system,
            self.yaw_system,
            self.rotor_blades,
            self.mechanical_brake,
            self.rotor_hub,
            self.gearbox,
            self.generator,
            self.supporting_structure,
            self.drive_train,
        ]

        self.env.log_action(
            agent=self.name,
            action="subassemblies created",
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
            Dictionary
        """

        self.transformer = Subassembly(
            self, self.env, "transformer", subassembly_data["transformer"]
        )

        self.subassemblies = [self.transformer]

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

    @property
    def operating_level(self) -> float:
        """The turbine's operating level, based on subassembly and cable performance.

        Returns
        -------
        float
            Operating level of the turbine.
        """
        if self.cable_failure:
            return 0.0
        else:
            return reduce(_product, [sub.operating_level for sub in self.subassemblies])

    def power(self, windspeed: Union[List[float], np.ndarray]) -> np.ndarray:
        """Generates the power output for an iterable of windspeed values.

        Parameters
        ----------
        windspeed : Union[List[float], np.ndarray]
            Windspeed values, in m/s.

        Returns
        -------
        np.ndarray
            Power production, in kW.
        """
        return self.power_curve(windspeed)
