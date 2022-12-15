"""Creates the Windfarm class/model."""
from __future__ import annotations

import csv
import logging
import datetime as dt
from math import fsum
from itertools import chain, combinations

import numpy as np
import pandas as pd  # type: ignore
import networkx as nx  # type: ignore
from geopy import distance  # type: ignore

from wombat.core import RepairManager, WombatEnvironment
from wombat.core.library import load_yaml
from wombat.windfarm.system import Cable, System
from wombat.utilities.utilities import cache


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

        # Set up the layout and instantiate all windfarm objects
        self._create_graph_layout(windfarm_layout)
        self._create_turbines_and_substations()
        self._create_cables()
        self.capacity: int | float = sum(
            self.system(turb).capacity for turb in self.turbine_id
        )
        self._create_substation_turbine_map()
        self.calculate_distance_matrix()

        # Create the logging items
        self.system_list = list(self.graph.nodes)
        # self._setup_logger()

        # Register the windfarm and start the logger
        self.repair_manager._register_windfarm(self)
        self.env._register_windfarm(self)
        # self.env.process(self._log_operations())

    def _create_graph_layout(self, windfarm_layout: str) -> None:
        """Creates a network layout of the windfarm start from the substation(s) to
        be able to capture downstream turbines that can be cut off in the event of a cable failure.

        Parameters
        ----------
        windfarm_layout : str
            Filename to use for reading in the windfarm layout; must be a csv file.
        """
        try:
            layout_path = str(self.env.data_dir / "project/plant" / windfarm_layout)
            layout = (
                pd.read_csv(layout_path)
                .sort_values(by=["string", "order"])
                .reset_index(drop=True)
            )
        except FileNotFoundError:
            layout_path = str(self.env.data_dir / "windfarm" / windfarm_layout)
            layout = (
                pd.read_csv(layout_path)
                .sort_values(by=["string", "order"])
                .reset_index(drop=True)
            )
            logging.warning(
                "DeprecationWarning: In v0.7, all wind farm layout files must be located in: '<library>/project/plant/"
            )
        layout.subassembly = layout.subassembly.fillna("")
        layout.upstream_cable = layout.upstream_cable.fillna("")

        windfarm = nx.DiGraph()
        windfarm.add_nodes_from(layout.id.values)

        # Assign the data attributes to the graph nodes
        for col in ("name", "latitude", "longitude", "subassembly"):
            nx.set_node_attributes(windfarm, dict(layout[["id", col]].values), name=col)

        # Determine which nodes are substations and which are turbines
        if "type" in layout.columns:
            if layout.loc[~layout.type.isin(("substation", "turbine"))].size > 0:
                raise ValueError(
                    "At least one value in the 'type' column are not one of 'substation' or 'turbine'."
                )
            substation_filter = layout.type == "substation"
            nx.set_node_attributes(
                windfarm, dict(layout[["id", "type"]].values), name="type"
            )
        else:
            substation_filter = layout.id == layout.substation_id
            _type = {True: "substation", False: "turbine"}
            d = {i: _type[val] for i, val in zip(layout.id, substation_filter.values)}
            nx.set_node_attributes(windfarm, d, name="type")

        self.substation_id = layout.loc[substation_filter, "id"].values
        self.turbine_id = layout.loc[~substation_filter, "id"].values
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
        for substation in self.substation_id:
            row = layout.loc[layout.id == substation]
            windfarm.add_edge(
                row.substation_id.values[0],
                substation,
                length=row.distance.values[0],
                cable=row.upstream_cable.values[0],
            )

        self.graph = windfarm

    def _create_turbines_and_substations(self) -> None:
        """Instantiates the turbine and substation models as defined in the
        user-provided layout file, and connects these models to the appropriate graph
        nodes to create a fully representative windfarm network model.

        Raises
        ------
        ValueError
            Raised if the subassembly data is not provided in the layout file.
        """
        bad_data_location_messages = []
        for system_id, data in self.graph.nodes(data=True):
            if data["subassembly"] == "":
                raise ValueError(
                    "A 'subassembly' file must be specified for all nodes in the windfarm layout!"
                )

            try:
                subassembly_dict = load_yaml(
                    self.env.data_dir / f"{data['type']}s", data["subassembly"]
                )
            except FileNotFoundError:
                subassembly_dict = load_yaml(
                    self.env.data_dir / "windfarm", data["subassembly"]
                )
                message = f"In v0.7, all {data['type']} configurations must be located in: '<library>/{data['type']}s"
                bad_data_location_messages.append(message)
            self.graph.nodes[system_id]["system"] = System(
                self.env,
                self.repair_manager,
                system_id,
                data["name"],
                subassembly_dict,
                data["type"],
            )

        # Raise the warning for soon-to-be deprecated library structure
        bad_data_location_messages = list(set(bad_data_location_messages))
        for message in bad_data_location_messages:
            logging.warning(f"DeprecationWarning: {message}")

    def _create_cables(self) -> None:
        """Instantiates the cable models as defined in the user-provided layout file,
        and connects these models to the appropriate graph edges to create a fully
        representative windfarm network model.

        Raises
        ------
        ValueError
            Raised if the cable model is not specified.
        """
        bad_data_location_messages = []
        for start_node, end_node, data in self.graph.edges(data=True):
            # Check that the cable data is provided
            if data["cable"] == "":
                raise ValueError(
                    "A 'cable' file must be specified for all nodes in the windfarm layout!"
                )
            try:
                cable_dict = load_yaml(self.env.data_dir / "cables", data["cable"])
            except FileNotFoundError:
                cable_dict = load_yaml(self.env.data_dir / "windfarm", data["cable"])
                bad_data_location_messages.append(
                    "In v0.7, all cable configurations must be located in: '<library>/cables/"
                )

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

            data["cable"] = Cable(
                self,
                self.env,
                start_node,
                end_node,
                cable_dict,
            )

            # Calaculate the geometric center point
            end_points = np.array((start_coordinates, end_coordinates))
            data["latitude"], data["longitude"] = end_points.mean(axis=0)

        # Raise the warning for soon-to-be deprecated library structure
        bad_data_location_messages = list(set(bad_data_location_messages))
        for message in bad_data_location_messages:
            logging.warning(f"DeprecationWarning: {message}")

    def calculate_distance_matrix(self) -> None:
        """Calculates the geodesic distance, in km, between all of the windfarm's nodes, e.g.,
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
                weights=np.array([self.system(t).capacity for t in s_t_map[s_id]])
                / self.capacity,
            )
            for s_id in s_t_map
        }
        self.substation_turbine_map = s_t_map

    def _setup_logger(self, initial: bool = True):
        self._log_columns = [
            "datetime",
            "env_datetime",
            "env_time",
        ] + self.system_list

        self.env._operations_writer = csv.DictWriter(
            self.env._operations_csv, delimiter="|", fieldnames=self._log_columns
        )
        if initial:
            self.env._operations_writer.writeheader()

    def _log_operations(self):
        """Logs the operational data for a simulation."""
        message = dict(
            datetime=dt.datetime.now(),
            env_datetime=self.env.simulation_time,
            env_time=self.env.now,
        )
        message.update(
            {system: self.system(system).operating_level for system in self.system_list}
        )
        self.env._operations_writer.writerow(message)

        HOURS = 1
        while True:
            yield self.env.timeout(HOURS)
            message = dict(
                datetime=dt.datetime.now(),
                env_datetime=self.env.simulation_time,
                env_time=self.env.now,
            )
            message.update(
                {
                    system: self.system(system).operating_level
                    for system in self.system_list
                }
            )
            self.env._operations_writer.writerow(message)

    @cache
    def system(self, system_id: str) -> System:
        """Convenience function to returns the desired `System` object for a turbine or
        substation in the windfarm.

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

    @cache
    def cable(self, cable_id: tuple[str, str] | str) -> Cable:
        """Convenience function to returns the desired `Cable` object for a cable in the
        windfarm.

        Parameters
        ----------
        cable_id : tuple[str, str] | str
            The cable's unique identifier, of the form: (``wombat.windfarm.System.id``,
            ``wombat.windfarm.System.id``), for the (downstream node id, upstream node id),
            or the ``Cable.id``.

        Returns
        -------
        Cable
            The ``Cable`` object.
        """
        if isinstance(cable_id, str):
            edge_id = tuple(cable_id.split("::")[1:])
        else:
            edge_id = cable_id
        try:
            return self.graph.edges[edge_id]["cable"]
        except KeyError:
            raise KeyError(f"Edge {edge_id} is invalid.")

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
                self.system(t).operating_level
                for t in self.substation_turbine_map[s_id]["turbines"]  # type: ignore
            ]
            for s_id in self.substation_turbine_map
        }
        availability = fsum(
            [
                self.system(s_id).operating_level
                * fsum(
                    operating_levels[s_id]
                    * self.substation_turbine_map[s_id]["weights"]  # type: ignore
                )
                for s_id in self.substation_turbine_map
            ]
        )
        return availability

    @property
    def current_availability_wo_servicing(self) -> float:
        """Calculates the product of all system ``operating_level`` variables across the
        windfarm using the following forumation, ignoring 0 operating level due to ongoing
        servicing.

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
                self.system(t).operating_level_wo_servicing
                for t in self.substation_turbine_map[s_id]["turbines"]  # type: ignore
            ]
            for s_id in self.substation_turbine_map
        }
        availability = fsum(
            [
                self.system(s_id).operating_level_wo_servicing
                * fsum(
                    operating_levels[s_id]
                    * self.substation_turbine_map[s_id]["weights"]  # type: ignore
                )
                for s_id in self.substation_turbine_map
            ]
        )
        return availability
