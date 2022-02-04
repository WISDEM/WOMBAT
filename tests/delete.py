from typing import Any, List, Union, Protocol
from functools import cached_property

from attrs import field, define


# from wombat.core import ServiceEquipment


@define(auto_attribs=True)
class EquipmentMap:
    """Internal mapping for servicing equipment strategy information."""

    strategy_threshold: Union[int, float] = field()
    equipment: Any = field()


@define(auto_attribs=True)
class StrategyMap:
    """Internal mapping for equipment capabilities and their data."""

    CTV: List[EquipmentMap] = field(factory=list)
    SCN: List[EquipmentMap] = field(factory=list)
    LCN: List[EquipmentMap] = field(factory=list)
    CAB: List[EquipmentMap] = field(factory=list)
    RMT: List[EquipmentMap] = field(factory=list)
    DRN: List[EquipmentMap] = field(factory=list)
    DSV: List[EquipmentMap] = field(factory=list)
    is_running: bool = field(default=False, init=False)

    def update(
        self, capability: str, threshold: Union[int, float], equipment: Any
    ) -> None:
        """A method to update the strategy mapping between capability types and the
        available ``ServiceEquipment`` objects.

        Parameters
        ----------
        capability : str
            The ``equipment``'s capability.
        threshold : Union[int, float]
            The threshold for ``equipment``'s strategy.
        equipment : ServiceEquipment
            The actual ``ServiceEquipment`` object to be logged.

        Raises
        ------
        ValueError
            Raised if there is an invalid capability, though this shouldn't be able to
            be reached.
        """
        if capability == "CTV":
            self.CTV.append(EquipmentMap(threshold, equipment))
        elif capability == "SCN":
            self.SCN.append(EquipmentMap(threshold, equipment))
        elif capability == "LCN":
            self.LCN.append(EquipmentMap(threshold, equipment))
        elif capability == "CAB":
            self.CAB.append(EquipmentMap(threshold, equipment))
        elif capability == "RMT":
            self.RMT.append(EquipmentMap(threshold, equipment))
        elif capability == "DRN":
            self.DRN.append(EquipmentMap(threshold, equipment))
        elif capability == "DSV":
            self.DSV.append(EquipmentMap(threshold, equipment))
        else:
            # This should not even be able to be reached
            raise ValueError("Invalid servicing equipment has been provided!")

        self.is_running = True


sm = StrategyMap()
print(sm)
print(sm.is_running)

sm.update("CTV", 0.9, 45)
print(sm)
print(sm.is_running)
