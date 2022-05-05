"""Creates the Port class that provies the tow-to-port repair capabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from simpy.resources.store import FilterStore, FilterStoreGet

from wombat.core.library import load_yaml
from wombat.core.environment import WombatEnvironment
from wombat.core.data_classes import PortConfig
from wombat.utilities.utilities import check_working_hours
from wombat.core.repair_management import RepairManager


try:
    from functools import cache  # type: ignore
except ImportError:
    from functools import lru_cache

    cache = lru_cache(None)


HOURS_IN_DAY = 24


class Port(FilterStore):
    def __init__(
        self,
        env: WombatEnvironment,
        config: str | Path,
        repair_manager: RepairManager,
        capacity: float = np.inf,
    ) -> None:
        super().__init__(env, capacity)

        self.env = env
        self.manager = RepairManager

        settings = load_yaml(env.data_dir / "repair", config)
        self.settings = PortConfig.from_dict(settings)

        self._check_working_hours()

    def _check_working_hours(self) -> None:
        """Checks the working hours of the port and overrides a default (-1) to
        the ``env`` settings, otherwise hours remain the same.
        """
        self.settings._set_environment_shift(
            *check_working_hours(
                self.env.workday_start,
                self.env.workday_end,
                self.settings.workday_start,
                self.settings.workday_end,
            )
        )

    def get_system_requests_from_manager(self, system_id: str) -> None:
        """Gets all of a given system's repair requests from the simulation's repair
        manager, removes them from that queue, and puts them in the port's queue.

        Parameters
        ----------
        system_id : str
            The ``System.id`` attribute from the system that will be repaired at port.
        """
        requests = self.manager.get_all_requests_for_system(
            self.settings.name, system_id
        )
        if requests is None:
            return

        self.items.extend(requests)
        for request in requests:
            self.env.log_action(
                system_id=request.system_id,
                system_name=request.system_name,
                part_id=request.subassembly_id,
                part_name=request.subassembly_name,
                system_ol=float("nan"),
                part_ol=float("nan"),
                agent=self.settings.name,
                action="requests moved to port",
                reason="at-port repair can now proceed",
                request_id=request.request_id,
            )
