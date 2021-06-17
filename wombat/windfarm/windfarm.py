"""Creates the Windfarm class/model."""

import logging  # type: ignore
import os  # type: ignore

import networkx as nx  # type: ignore
import pandas as pd  # type: ignore
from geopy import distance  # type: ignore

from wombat.core import RepairManager, WombatEnvironment
from wombat.utilities import load_yaml
from wombat.windfarm.system import Cable, System


class Windfarm:
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
        layout_path = os.path.join(self.env.data_dir, "windfarm", windfarm_layout)

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
                os.path.join(self.env.data_dir, "windfarm"), data["subassembly"]
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

            # If the real distance/cable length is not input, then the geodesic distance is calculated
            if data["length"] == 0:
                start_coordinates = (
                    self.graph.nodes[start_node]["latitude"],
                    self.graph.nodes[start_node]["longitude"],
                )
                end_coordinates = (
                    self.graph.nodes[end_node]["latitude"],
                    self.graph.nodes[end_node]["longitude"],
                )
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
        pass

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
            [
                self.graph.nodes[system]["system"].operating_level
                for system in system_list
            ]
        )
        message = " :: ".join((f"{m}" for m in message))
        self.env._operations_logger.info(message)

        HOURS = 1
        while True:
            yield self.env.timeout(HOURS)
            message = [self.env.simulation_time, self.env.now] + [
                self.node_system(system).operating_level for system in system_list
            ]
            message = " :: ".join(f"{m}" for m in message)
            self.env._operations_logger.info(message)

    def node_system(self, system_id: str) -> System:
        """Returns the desired node in the windfarm.

        Parameters
        ----------
        system_id : str
            The system's unique identifier, `wombat.windfarm.System.id`.

        Returns
        -------
        System
            The `System` object.
        """
        return self.graph.nodes[system_id]["system"]
