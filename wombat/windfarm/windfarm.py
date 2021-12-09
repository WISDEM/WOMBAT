"""Creates the Windfarm class/model."""

import os  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import logging  # type: ignore
import networkx as nx  # type: ignore
from math import fsum
from geopy import distance  # type: ignore
from itertools import chain, combinations

from wombat.core import RepairManager, WombatEnvironment
from wombat.core.library import load_yaml
from wombat.windfarm.system import Cable, System


class Windfarm:
    """The primary class for operating on objects within a windfarm. The substations,
    cables, and turbines are created as a network object to be more appropriately accessed
    and controlled.
    """

    def __init__(
        self,
        env: WombatEnvironment,
        windfarm_layout: str,
        repair_manager: RepairManager,
    ) -> None:

        self.env = env
        self.repair_manager = repair_manager

        # self._logging_setup()

        self._create_graph_layout(windfarm_layout)
        self._create_turbines_and_substations()
        self._create_cables()
        self.capacity = sum(self.node_system(turb).capacity for turb in self.turbine_id)
        self._create_substation_turbine_map()
        self.calculate_distance_matrix()

        self.repair_manager._register_windfarm(self)

        self.env.process(self._log_operations())

    def _logging_setup(self) -> None:
        """Completes the setup for logging data.

        Parameters
        ----------
        which : str
            One of "events" or "operations". For creating event logs or operational
            status logs.
        """

        logging.basicConfig(
            format="%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s",
            filename=self.env.operations_log_fname,
            filemode="w",
            level=logging.DEBUG,
        )
        self._operations_logger = logging.getLogger(__name__)

    def _create_graph_layout(self, windfarm_layout: str) -> None:
        """Creates a network layout of the windfarm start from the substation(s) to
        be able to capture downstream turbines that can be cut off in the event of a cable failure.

        Parameters
        ----------
        windfarm_layout : str
            Filename to use for reading in the windfarm layout; must be a csv file.
        """
        layout_path = str(self.env.data_dir / "windfarm" / windfarm_layout)
        layout = (
            pd.read_csv(layout_path)
            .sort_values(by=["string", "order"])
            .reset_index(drop=True)
        )
        layout.subassembly = layout.subassembly.fillna("")

        windfarm = nx.DiGraph()
        windfarm.add_nodes_from(layout.id.values)

        # Assign the data attributes to the graph nodes
        for col in ("name", "latitude", "longitude", "subassembly"):
            d = {i: n for i, n in layout[["id", col]].values}
            nx.set_node_attributes(windfarm, d, name=col)

        # Determine which nodes are substations and which are turbines
        substation_filter = layout.id == layout.substation_id
        self.substation_id = layout[substation_filter].id.unique()
        self.turbine_id = layout[~substation_filter].id.unique()
        _type = {True: "substation", False: "turbine"}
        d = {i: _type[val] for i, val in zip(layout.id, substation_filter.values)}
        nx.set_node_attributes(windfarm, d, name="type")

        substations = layout[substation_filter].copy()
        turbines = layout[~substation_filter].copy()
        substation_sections = [
            turbines[turbines.substation_id == substation]
            for substation in substations.id
        ]
        for section in substation_sections:
            for _, row in section.iterrows():
                if row.order == 0:
                    start: str = row.substation_id
                else:
                    start = current  # noqa: F821
                current: str = row.id
                windfarm.add_edge(
                    start, current, length=row.distance, cable=row.upstream_cable
                )

        self.graph = windfarm

    def _create_turbines_and_substations(self) -> None:
        for system_id, data in self.graph.nodes(data=True):
            if data["subassembly"] == "":
                raise ValueError(
                    "A 'subassembly' file must be specified for all nodes in the windfarm layout!"
                )

            subassembly_dict = load_yaml(
                self.env.data_dir / "windfarm", data["subassembly"]
            )
            self.graph.nodes[system_id]["system"] = System(
                self.env,
                self.repair_manager,
                system_id,
                data["name"],
                subassembly_dict,
                data["type"],
            )

    def _create_cables(self) -> None:
        for start_node, end_node, data in self.graph.edges(data=True):

            start_coordinates = (
                self.graph.nodes[start_node]["latitude"],
                self.graph.nodes[start_node]["longitude"],
            )
            end_coordinates = (
                self.graph.nodes[end_node]["latitude"],
                self.graph.nodes[end_node]["longitude"],
            )

            # If the real distance/cable length is not input, then the geodesic distance
            # is calculated
            if data["length"] == 0:
                data["length"] = distance.geodesic(
                    start_coordinates, end_coordinates, ellipsoid="WGS-84"
                ).km

            upstream_turbines = nx.dfs_successors(self.graph, end_node)
            cable_dict = load_yaml(
                os.path.join(self.env.data_dir, "windfarm"), data["cable"]
            )
            data["cable"] = Cable(
                self,
                self.env,
                f"cable::{start_node}::{end_node}",
                start_node,
                upstream_turbines,
                cable_dict,
            )

            # Calaculate the geometric center point
            end_points = np.array((start_coordinates, end_coordinates))
            data["latitude"], data["longitude"] = end_points.mean(axis=0)

    def calculate_distance_matrix(self) -> None:
        """Calculates hte geodesic distance, in km, between all of the windfarm's nodes, e.g.,
        substations and turbines, and cables.
        """
        ids = list(self.graph.nodes())
        ids.extend([data["cable"].id for *_, data in self.graph.edges(data=True)])
        coords = [
            (data["latitude"], data["longitude"])
            for *_, data in (*self.graph.nodes(data=True), *self.graph.edges(data=True))
        ]

        dist = [distance.geodesic(c1, c2).km for c1, c2 in combinations(coords, 2)]
        dist_arr = np.ones((len(ids), len(ids)))
        triangle_ix = np.triu_indices_from(dist_arr, 1)
        dist_arr[triangle_ix] = dist_arr.T[triangle_ix] = dist

        # Set the self distance to infinity, so that only one crew can be dropped off
        # at a single point
        np.fill_diagonal(dist_arr, np.inf)

        self.distance_matrix = pd.DataFrame(dist_arr, index=ids, columns=ids)

    def _create_substation_turbine_map(self) -> None:
        """Creates ``substation_turbine_map``, a dictionary, that maps substation(s) to
        the dependent turbines in the windfarm, and the weighting of each turbine in the
        windfarm.
        """
        # Get all turbines dependent on each substation
        s_t_map = {
            s_id: list(
                chain.from_iterable(nx.dfs_successors(self.graph, source=s_id).values())
            )
            for s_id in self.substation_id
        }

        # Reorient the mapping to have the turbine list and the capacity-based weighting
        # of each turbine
        s_t_map = {
            s_id: dict(  # type: ignore
                turbines=np.array(s_t_map[s_id]),
                weights=np.array([self.node_system(t).capacity for t in s_t_map[s_id]])
                / self.capacity,
            )
            for s_id in s_t_map
        }
        self.substation_turbine_map = s_t_map

    def _log_operations(self):
        """Logs the operational data for a simulation."""
        system_list = list(self.graph.nodes)
        columns = [
            "datetime",
            "name",
            "level",
            "env_datetime",
            "env_time",
        ] + system_list
        self.env._operations_logger.info(" :: ".join(columns))

        message = [self.env.simulation_time, self.env.now]
        message.extend(
            [self.node_system(system).operating_level for system in system_list]
        )
        message = " :: ".join((f"{m}" for m in message))  # type: ignore
        self.env._operations_logger.info(message)

        HOURS = 1
        while True:
            yield self.env.timeout(HOURS)
            message = [self.env.simulation_time, self.env.now] + [
                self.node_system(system).operating_level for system in system_list
            ]
            message = " :: ".join(f"{m}" for m in message)  # type: ignore
            self.env._operations_logger.info(message)

    def node_system(self, system_id: str) -> System:
        """Returns the desired node in the windfarm.

        Parameters
        ----------
        system_id : str
            The system's unique identifier, ``wombat.windfarm.System.id``.

        Returns
        -------
        System
            The ``System`` object.
        """
        return self.graph.nodes[system_id]["system"]

    @property
    def current_availability(self) -> float:
        """Calculates the product of all system ``operating_level`` variables across the
        windfarm using the following forumation

        .. math::
            \sum{
                OperatingLevel_{substation_{i}} *
                \sum{OperatingLevel_{turbine_{j}} * Weight_{turbine_{j}}}
            }

        where the :math:``{OperatingLevel}`` is the product of the operating level
        of each subassembly on a given system (substation or turbine), and the
        :math:``{Weight}`` is the proportion of one turbine's capacity relative to
        the whole windfarm.
        """  # noqa: W605
        operating_levels = {
            s_id: [
                self.node_system(t).operating_level
                for t in self.substation_turbine_map[s_id]["turbines"]  # type: ignore
            ]
            for s_id in self.substation_turbine_map
        }
        availability = fsum(
            [
                self.node_system(s_id).operating_level
                * fsum(
                    operating_levels[s_id]
                    * self.substation_turbine_map[s_id]["weights"]  # type: ignore
                )
                for s_id in self.substation_turbine_map
            ]
        )
        return availability
