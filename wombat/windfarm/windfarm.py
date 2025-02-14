"""Creates the Windfarm class/model."""

from __future__ import annotations

import csv
import datetime as dt
import itertools
from math import fsum
from functools import cache

import numpy as np
import pandas as pd
import networkx as nx
from geopy import distance

from wombat.core import RepairManager, WombatEnvironment
from wombat.core.library import load_yaml
from wombat.windfarm.system import Cable, System
from wombat.core.data_classes import String, SubString, WindFarmMap, SubstationMap


class Windfarm:
    """The primary class for operating on objects within a windfarm. The substations,
    cables, and turbines are created as a network object to be more appropriately
    accessed and controlled.
    """

    def __init__(
        self,
        env: WombatEnvironment,
        windfarm_layout: str,
        repair_manager: RepairManager,
        substations: dict[str, dict] | None = None,
        turbines: dict[str, dict] | None = None,
        cables: dict[str, dict] | None = None,
    ) -> None:
        self.env = env
        self.repair_manager = repair_manager

        # Set up the layout and instantiate all windfarm objects
        self._inputs: dict[str, dict | None] = {
            "turbine": turbines,
            "substation": substations,
            "cable": cables,
        }
        self.configs: dict[str, dict] = {"turbine": {}, "substation": {}, "cable": {}}
        self._create_graph_layout(windfarm_layout)
        self._create_turbines_and_substations()
        self._create_cables()
        self.capacity: int | float = sum(
            self.system(turb).capacity for turb in self.turbine_id
        )
        self._create_substation_turbine_map()
        self._create_wind_farm_map()
        self.finish_setup()
        self.calculate_distance_matrix()

        # Create the logging items
        self.system_list = list(self.graph.nodes)
        self._setup_logger()

        # Register the windfarm and start the logger
        self.repair_manager._register_windfarm(self)
        self.env._register_windfarm(self)
        self.env.process(self._log_operations())

    def _create_graph_layout(self, windfarm_layout: str) -> None:
        """Creates a network layout of the windfarm start from the substation(s) to
        be able to capture downstream turbines that can be cut off in the event of a
        cable failure.

        Parameters
        ----------
        windfarm_layout : str
            Filename to use for reading in the windfarm layout; must be a csv file.
        """
        # Read in the layout CSV file, then sort it by string, then order to ensure
        # it can be traversed in sequential order later
        layout_path = str(self.env.data_dir / "project/plant" / windfarm_layout)
        layout = (
            pd.read_csv(layout_path)
            .sort_values(by=["string", "order"])
            .reset_index(drop=True)
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
            # Extract the type directly from the layout file
            if layout.loc[~layout.type.isin(("substation", "turbine"))].size > 0:
                raise ValueError(
                    "At least one value in the 'type' column are not one of:"
                    " 'substation' or 'turbine'."
                )
            substation_filter = layout.type == "substation"
            nx.set_node_attributes(
                windfarm, dict(layout[["id", "type"]].values), name="type"
            )
        else:
            # Deduce substations by their self-connected setting for interconnection
            substation_filter = layout.id == layout.substation_id
            _type = {True: "substation", False: "turbine"}
            d = {i: _type[val] for i, val in zip(layout.id, substation_filter.values)}
            nx.set_node_attributes(windfarm, d, name="type")

        self.turbine_id: np.ndarray = layout.loc[~substation_filter, "id"].values

        self.substation_id = layout.loc[substation_filter, "id"].values
        for substation in self.substation_id:
            windfarm.nodes[substation]["connection"] = layout.loc[
                layout.id == substation, "substation_id"
            ].values[0]

        # Create a mapping for each substation to all connected turbines (subgraphs)
        substations = layout[substation_filter].copy()
        turbines = layout[~substation_filter].copy()
        substation_sections = [
            turbines[turbines.substation_id == substation]
            for substation in substations.id
        ]

        # For each subgraph, create the edge connections between substations and
        # turbines. Note: these are pre-sorted in the layout creation step.
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
        # Create the substation to substation and substation to self connections
        for substation in self.substation_id:
            row = layout.loc[layout.id == substation]
            windfarm.add_edge(
                row.substation_id.values[0],
                substation,
                length=row.distance.values[0],
                cable=row.upstream_cable.values[0],
            )

        self.graph: nx.DiGraph = windfarm
        self.layout_df = layout

    def _create_turbines_and_substations(self) -> None:
        """Instantiates the turbine and substation models as defined in the
        user-provided layout file, and connects these models to the appropriate graph
        nodes to create a fully representative windfarm network model.

        Raises
        ------
        ValueError
            Raised if the subassembly data is not provided in the layout file.
        """
        # Loop through all nodes in the graph, and create the actual simulation objects
        for system_id, data in self.graph.nodes(data=True):
            name = data["subassembly"]
            node_type = data["type"]
            if name == "":
                raise ValueError(
                    "A 'subassembly' file must be specified for all nodes in the"
                    " windfarm layout!"
                )

            # Read in unique system configuration files only once, and reference
            # the existing dictionary when possible to reduce I/O
            if (subassembly_dict := self.configs[node_type].get(name)) is None:
                if name.endswith((".yml", ".yaml")):
                    subassembly_dict = load_yaml(
                        self.env.data_dir / f"{node_type}s", name
                    )
                else:
                    if self._inputs[node_type] is None:
                        msg = (
                            f"{node_type} configuration: '{name}' is not a file, and"
                            f" no input to {node_type}s was provided."
                        )
                        raise ValueError(msg)
                    if (subassembly_dict := self._inputs[node_type].get(name)) is None:  # type: ignore
                        raise KeyError(
                            f"No configuration provided for {node_type}: {name}"
                        )
                self.configs[node_type][name] = subassembly_dict

            # Create the turbine or substation simulation object
            self.graph.nodes[system_id]["system"] = System(
                self.env,
                self.repair_manager,
                system_id,
                data["name"],
                self.configs[node_type][name],
                node_type,
            )

    def _create_cables(self) -> None:
        """Instantiates the cable models as defined in the user-provided layout file,
        and connects these models to the appropriate graph edges to create a fully
        representative windfarm network model.

        Raises
        ------
        ValueError
            Raised if the cable model is not specified.
        """
        get_name = "upstream_cable_name" in self.layout_df

        # Loop over all the edges in the graph and create cable objects
        for start_node, end_node, data in self.graph.edges(data=True):
            name = data["cable"]

            # Check that the cable data is provided
            if name == "":
                raise ValueError(
                    "An 'upstream_cable' file must be specified for all nodes in the"
                    " windfarm layout!"
                )

            # Read in unique cable configuration files once to reduce I/O
            if (cable_dict := self.configs["cable"].get(name)) is None:
                if name.endswith((".yml", ".yaml")):
                    cable_dict = load_yaml(self.env.data_dir / "cables", name)
                else:
                    if self._inputs["cables"] is None:
                        msg = (
                            f"cable configuration: '{name}' is not a file, and"
                            f" no configuration input to 'cables' was provided."
                        )
                        raise ValueError(msg)
                    if (cable_dict := self._inputs["cable"].get(name)) is None:  # type: ignore
                        raise KeyError(f"No configuration provided for 'cables: {name}")

                self.configs["cable"][name] = cable_dict

            # Get the lat, lon pairs for the start and end points
            start_coordinates = (
                self.graph.nodes[start_node]["latitude"],
                self.graph.nodes[start_node]["longitude"],
            )
            end_coordinates = (
                self.graph.nodes[end_node]["latitude"],
                self.graph.nodes[end_node]["longitude"],
            )

            # Get the unique naming of the cable connection if it's configured
            name = None
            if get_name:
                name, *_ = self.layout_df.loc[
                    self.layout_df.id == end_node, "upstream_cable_name"
                ]

            # If the real distance/cable length is not input, then the geodesic distance
            # is calculated
            if data["length"] == 0:
                data["length"] = distance.geodesic(
                    start_coordinates, end_coordinates, ellipsoid="WGS-84"
                ).km

            # Encode whether it is an array cable or an export cable
            if self.graph.nodes[end_node]["type"] == "substation":
                data["type"] = "export"
            else:
                data["type"] = "array"

            # Create the Cable simulation object
            data["cable"] = Cable(
                self, self.env, data["type"], start_node, end_node, cable_dict, name
            )

            # Calaculate the geometric center point of the cable for later
            # determining travel distances to cables
            end_points = np.array((start_coordinates, end_coordinates))
            data["latitude"], data["longitude"] = end_points.mean(axis=0)

    def calculate_distance_matrix(self) -> None:
        """Calculates the geodesic distance, in km, between all of the windfarm's nodes,
        e.g., substations and turbines, and cables.
        """
        ids = list(self.graph.nodes())
        ids.extend([data["cable"].id for *_, data in self.graph.edges(data=True)])
        coords = [
            (data["latitude"], data["longitude"])
            for *_, data in (*self.graph.nodes(data=True), *self.graph.edges(data=True))
        ]

        dist = [
            distance.geodesic(c1, c2).km for c1, c2 in itertools.combinations(coords, 2)
        ]
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
        # Get all turbines connected to each substation, excepting any connected via
        # export cables that connect substations as these operate independently
        s_t_map: dict = {s: {"turbines": [], "weights": []} for s in self.substation_id}
        for substation_id in self.substation_id:
            nodes = set(
                nx.bfs_tree(self.graph, substation_id, depth_limit=1).nodes
            ).difference(self.substation_id)
            for node in list(nodes):
                nodes.update(
                    list(itertools.chain(*nx.dfs_successors(self.graph, node).values()))
                )
            s_t_map[substation_id]["turbines"] = np.array(list(nodes))

        # Reorient the mapping to have the turbine list and the capacity-based weighting
        # of each turbine
        for s_id in s_t_map:
            s_t_map[s_id]["weights"] = (
                np.array([self.system(t).capacity for t in s_t_map[s_id]["turbines"]])
                / self.capacity
            )

        self.substation_turbine_map: dict[str, dict[str, np.ndarray]] = s_t_map

        # Calculate the turbine weights
        self.turbine_weights: pd.DataFrame = (
            pd.concat([pd.DataFrame(val) for val in s_t_map.values()])
            .set_index("turbines")
            .T
        )

    def _create_wind_farm_map(self) -> None:
        """Creates a secondary graph object strictly for traversing the windfarm to turn
        on/off the appropriate turbines, substations, and cables more easily.
        """
        substations = self.substation_id
        graph = self.graph
        wind_map = dict(zip(substations, itertools.repeat({})))

        export = [el for el in graph.edges if el[1] in substations]
        for cable_tuple in export:
            self.cable(cable_tuple).set_string_details(*cable_tuple[::-1])

        for s_id in self.substation_id:
            start_nodes = list(
                set(nx.bfs_tree(graph, s_id, depth_limit=1).nodes).difference(
                    substations
                )
            )
            wind_map[s_id] = {"strings": dict(zip(start_nodes, itertools.repeat({})))}

            for start_node in start_nodes:
                upstream = list(
                    itertools.chain(*nx.dfs_successors(graph, start_node).values())
                )
                wind_map[s_id]["strings"][start_node] = {
                    start_node: SubString(
                        downstream=s_id,  # type: ignore
                        upstream=upstream,  # type: ignore
                    )
                }
                self.cable((s_id, start_node)).set_string_details(start_node, s_id)

                downstream = start_node
                for node in upstream:
                    wind_map[s_id]["strings"][start_node][node] = SubString(
                        downstream=downstream,  # type: ignore
                        upstream=list(  # tye: ignore
                            itertools.chain(*nx.dfs_successors(graph, node).values())
                        ),
                    )
                    self.cable((downstream, node)).set_string_details(start_node, s_id)
                    downstream = node

                wind_map[s_id]["strings"][start_node] = String(  # type: ignore
                    start=start_node, upstream_map=wind_map[s_id]["strings"][start_node]
                )
            wind_map[s_id] = SubstationMap(  # type: ignore
                string_starts=start_nodes,
                string_map=wind_map[s_id]["strings"],
                downstream=graph.nodes[s_id]["connection"],
            )
        self.wind_farm_map = WindFarmMap(
            substation_map=wind_map,
            export_cables=export,  # type: ignore
        )

    def finish_setup(self) -> None:
        """Final initialization hook for any substations, turbines, or cables."""
        for start_node, end_node in self.graph.edges():
            self.cable((start_node, end_node)).finish_setup()

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
        message = {
            "datetime": dt.datetime.now(),
            "env_datetime": self.env.simulation_time,
            "env_time": self.env.now,
        }
        message.update(
            {system: self.system(system).operating_level for system in self.system_list}
        )
        self.env._operations_writer.writerow(message)

        HOURS = 1
        while True:
            # Loop 10K times, to limit the number of times we write to the operations
            # log file. 10K was a crude optimization decision, so performance can vary
            # dependending on the simulation
            for _ in range(10000):
                yield self.env.timeout(HOURS)
                message = {
                    "datetime": dt.datetime.now(),
                    "env_datetime": self.env.simulation_time,
                    "env_time": self.env.now,
                }
                message.update(
                    {
                        system: self.system(system).operating_level
                        for system in self.system_list
                    }
                )
                self.env._operations_buffer.append(message)
            self.env._operations_writer.writerows(self.env._operations_buffer)
            self.env._operations_buffer.clear()

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
            ``wombat.windfarm.System.id``), for the (downstream node id, upstream node
            id), or the ``Cable.id``.

        Returns
        -------
        Cable
            The ``Cable`` object.
        """
        if isinstance(cable_id, str):
            edge_id = tuple(cable_id.split("::"))
            if len(edge_id) == 3:
                edge_id = edge_id[1:]
        else:
            edge_id = cable_id
        try:
            return self.graph.edges[edge_id]["cable"]
        except KeyError:
            raise KeyError(f"Edge {edge_id} is invalid.")

    @property
    def current_availability(self) -> float:
        r"""Calculates the product of all system ``operating_level`` variables across
        the windfarm using the following forumation.

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
        r"""Calculates the product of all system ``operating_level`` variables across
        the windfarm using the following forumation, ignoring 0 operating level due to
        ongoing servicing.

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
