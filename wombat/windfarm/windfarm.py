"""Creates the Windfarm class/model."""
from __future__ import annotations

import csv
import datetime as dt
import itertools
from math import fsum


import numpy as np
import pandas as pd
import networkx as nx
from geopy import distance

from wombat.core import  WombatEnvironment, RepairManager
from wombat.core.library import load_yaml
from wombat.windfarm.system import Cable, System, Mooring 
from wombat.core.data_classes import String, SubString, Anchor, MooringLine, WindFarmMap, SubstationMap, MooringMap
from wombat.utilities.utilities import cache


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
        anchor_layout: str | None = None, 
        ) -> None:
        self.env = env
        self.repair_manager = repair_manager

        # Set up the layout and instantiate all windfarm objects
        self.configs: dict[str, dict] = {"turbine": {}, "substation": {}, "cable": {}, "mooringline": {}, "anchor":{}}
        
        if anchor_layout is not None:
            self.anchor_layout = anchor_layout
            
        self._create_graph_layout(windfarm_layout)
        self._create_turbines_and_substations()
        self._create_cables()
        if anchor_layout is not None:
            self._create_mooring_layout(anchor_layout)
        self.capacity: int | float = sum(
            self.system(turb).capacity for turb in self.turbine_id
        )
        
       

        self._create_substation_turbine_map()
        self._create_wind_farm_map()
       
        if anchor_layout is not None:
            self._create_mooring_map()
            self.integrate_mooring_with_main_graph()  
        

            self.create_combo_farm()
       
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
            substation_filter = layout.id == layout.substation_id
            _type = {True: "substation", False: "turbine"}
            d = {i: _type[val] for i, val in zip(layout.id, substation_filter.values)}
            nx.set_node_attributes(windfarm, d, name="type")

        self.substation_id = layout.loc[substation_filter, "id"].values
        for substation in self.substation_id:
            windfarm.nodes[substation]["connection"] = layout.loc[
                layout.id == substation, "substation_id"
            ].values[0]

        self.turbine_id: np.ndarray = layout.loc[~substation_filter, "id"].values
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
        for system_id, data in self.graph.nodes(data=True):
            name = data["subassembly"]
            node_type = data["type"]
            if name == "":
                raise ValueError(
                    "A 'subassembly' file must be specified for all nodes in the"
                    " windfarm layout!"
                )

            if (subassembly_dict := self.configs[node_type].get(name)) is None:
                subassembly_dict = load_yaml(self.env.data_dir / f"{node_type}s", name)
                self.configs[node_type][name] = subassembly_dict

            self.graph.nodes[system_id]["system"] = System(
                self.env,
                self.repair_manager,
                system_id,
                data["name"],
                subassembly_dict,
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
        bad_data_location_messages = []
        for start_node, end_node, data in self.graph.edges(data=True):
            # Skip cable setup if either node is an anchor
            if self.is_anchor(start_node) or self.is_anchor(end_node):
                continue  # Skip this iteration, effectively ignoring the edge for cable creation
    
           
            
            name = data["cable"]

            # Check that the cable data is provided
            if name == "":
                raise ValueError(
                    "An 'upstream_cable' file must be specified for all nodes in the"
                    " windfarm layout!"
                )
            cable_dict = self.configs["cable"].get(name, None)
            if (cable_dict := self.configs["cable"].get(name)) is None:
                try:
                    cable_dict = load_yaml(self.env.data_dir / "cables", data["cable"])
                except FileNotFoundError:
                    cable_dict = load_yaml(
                        self.env.data_dir / "windfarm", data["cable"]
                    )
                    bad_data_location_messages.append(
                        "In v0.7, all cable configurations must be located in:"
                        " '<library>/cables/"
                    )
                self.configs["cable"][name] = cable_dict

            start_coordinates = (
                self.graph.nodes[start_node]["latitude"],
                self.graph.nodes[start_node]["longitude"],
            )
            end_coordinates = (
                self.graph.nodes[end_node]["latitude"],
                self.graph.nodes[end_node]["longitude"],
            )

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

            if self.graph.nodes[end_node]["type"] == "substation":
                data["type"] = "export"
            else:
                data["type"] = "array"

            data["cable"] = Cable(
                self, self.env, data["type"], start_node, end_node, cable_dict, name
            )

            # Calaculate the geometric center point
            end_points = np.array((start_coordinates, end_coordinates))
            data["latitude"], data["longitude"] = end_points.mean(axis=0)
    

    

    def _create_mooring_layout(self, anchor_layout: str) -> None:
        if not hasattr(self, 'mooring_graph'):
            self.mooring_graph = nx.Graph()
    
        # Load mooring layout data from CSV
        mooring_layout_path = self.env.data_dir / "project/plant" / anchor_layout
        mooring_layout_df = pd.read_csv(mooring_layout_path)
    

        # Dictionary to store Mooring objects and track anchors
        anchor_dict = {}

        # Process each row in the DataFrame
        for _, row in mooring_layout_df.iterrows():
            anchor_id = row['anchor_id']
            anchor_type = row['type']
            anchor_coords = (row['latitude'], row['longitude'])
    
    
            if anchor_id in anchor_dict:
                # Update the existing anchor's connected systems
                existing_anchor = anchor_dict[anchor_id]
                existing_anchor['connected_systems'].extend(
                    [cs.strip() for cs in row['All Connected Items'].split(',') if cs.strip()]
                )
            else:
                # Create a new anchor
                yaml_filename = f"corewind_{anchor_type}.yaml"
                anchor_yaml_data = load_yaml(self.env.data_dir / "mooring", yaml_filename)
    
                # Ensure all required keys ('name', 'maintenance', 'failures') are present
                if not all(key in anchor_yaml_data for key in ['name', 'maintenance', 'failures']):
                    raise ValueError(f"Configuration for anchor {anchor_id} is missing required fields.")
    
                connected_systems = [cs.strip() for cs in row['All Connected Items'].split(',') if cs.strip()]
    
                mooring_data = {
                    'anchor': anchor_yaml_data,
                    'mooringlines': {}
                }
    
                mooringline_ids = []
                for i in range(1, len(connected_systems) + 1):
                    connected_item_key = f'Connected Item {i}'
                    mooringline_id_key = f'MooringLine ID {i}'
                    mooringline_type_key = f'Mooring Type {i}'
    
                    connected_system = row[connected_item_key].strip()
                    mooringline_id = row[mooringline_id_key].strip()
                    mooringline_type = row.get(mooringline_type_key, 'default').strip()
    
                    mooring_yaml_filename = f"{mooringline_type}"
                    mooring_yaml_data = load_yaml(self.env.data_dir / "mooring", mooring_yaml_filename)
                    mooring_data['mooringlines'][mooringline_id] = mooring_yaml_data
    
                    mooringline_ids.append(mooringline_id)
    
                mooring = Mooring(
                    windfarm=self,
                    env=self.env,
                    anchor_id=anchor_id,
                    mooringline_ids=mooringline_ids,
                    connected_systems=connected_systems,
                    anchor_data=anchor_yaml_data,
                    mooringlines_data=mooring_data['mooringlines'],
                    name=None
                )
    
                self.mooring_graph.add_node(
                    anchor_id,
                    type='anchor',
                    anchor_type=anchor_type,
                    latitude=anchor_coords[0],
                    longitude=anchor_coords[1],
                    connected_systems=connected_systems,
                    config=anchor_yaml_data,
                    object=mooring
                )
    
                anchor_dict[anchor_id] = {'mooring': mooring, 'connected_systems': connected_systems}
                
    
            for connected_system in connected_systems:
                system_coords = (self.graph.nodes[connected_system]['latitude'], self.graph.nodes[connected_system]['longitude'])
                center_coords = [(anchor_coords[0] + system_coords[0]) / 2, (anchor_coords[1] + system_coords[1]) / 2]
    
                mooringline_id = mooringline_ids.pop(0)
                self.mooring_graph.add_edge(
                    anchor_id, connected_system,
                    mooringline_id=mooringline_id,
                    type=mooringline_type,
                    center_coords=center_coords,
                    config=mooring_yaml_data,
                    maintenance=mooring_yaml_data.get('maintenance', []),
                    failures=mooring_yaml_data.get('failures', [])
                )
    
        # Log successful setup
        #print(f"Mooring layout for {len(mooring_layout_df)} anchors has been created with all associated mooring lines.")
    
        # Store the moorings dictionary in the class attribute
        self.moorings = {anchor_id: anchor['mooring'] for anchor_id, anchor in anchor_dict.items()}


   
    
    def is_anchor(self, node_id: str) -> bool:
        """Check if a node ID represents an anchor."""
        return "anchor" in node_id

    def _is_turbine(self, node_id: str) -> bool:
        """Check if a node ID represents a turbine."""
        return node_id.startswith("WTG")

    def _is_substation(self, node_id: str) -> bool:
        """Check if a node ID represents a substation."""
        return node_id.startswith("SS")
    
   
 
    def calculate_distance_matrix(self) -> None:
        """Calculates the geodesic distance, in km, between all nodes (substations, turbines, anchors) and midpoints of edges (cables, mooring lines)."""
        
        # Collect coordinates for substations, turbines, and anchors
        coords = {node_id: (data["latitude"], data["longitude"]) for node_id, data in self.graph.nodes(data=True) if "latitude" in data and "longitude" in data}
        if hasattr(self, "anchor_layout"):
            coords.update({node_id: (data["latitude"], data["longitude"]) for node_id, data in self.mooring_graph.nodes(data=True) if "latitude" in data and "longitude" in data})
        
        # Calculate midpoints for cables and mooring lines and treat them as virtual nodes
        for start_node, end_node, data in self.graph.edges(data=True):
            if "latitude" in data and "longitude" in data:  # If direct coordinates are provided
                midpoint_id = f"{start_node}::{end_node}"
                coords[midpoint_id] = (data["latitude"], data["longitude"])
        
        if hasattr(self, "anchor_layout"):
            for start_node, end_node, data in self.mooring_graph.edges(data=True):
                if "latitude" in data and "longitude" in data:  # If direct coordinates are provided
                    midpoint_id = f"{start_node}--{end_node}"  # Use '--' for mooring lines
                    coords[midpoint_id] = (data["latitude"], data["longitude"])
            
            
        # Calculate distances
        ids = list(coords.keys())
        dist_arr = np.zeros((len(ids), len(ids)))
        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids):
                if i != j:
                    dist_arr[i, j] = distance.geodesic(coords[id1], coords[id2]).km
                else:
                    dist_arr[i, j] = np.inf  # Set distance to self as infinity

        # Create the distance matrix DataFrame
        self.distance_matrix = pd.DataFrame(dist_arr, index=ids, columns=ids)


    def get_systems(self, system_ids: list[str]):
        """Fetches multiple System objects based on their IDs.
        
        Parameters
        ----------
        system_ids : list[str]
            A list of system IDs for which to fetch the System objects.
            
        Returns
        -------
        list[System]
            A list of System objects corresponding to the provided IDs.
        """
        return [self.system(system_id) for system_id in system_ids]    

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
        
    def _create_mooring_map(self):
        if not hasattr(self, 'mooring_map'):
            self.mooring_map = MooringMap()
        
        # Iterate through the nodes in the mooring_graph, adding anchors to the mooring map
        for node_id, node_attrs in self.mooring_graph.nodes(data=True):
            if 'anchor' in node_attrs.get('type', '').lower():
                # Create an Anchor object from the node attributes
                anchor = Anchor(
                    id=node_id,
                    type=node_attrs['anchor_type'],
                    coordinates=(node_attrs['longitude'], node_attrs['latitude'])
                )
    

            
                # Add connections to the anchor object
                for connected_system in node_attrs['connected_systems']:
                    anchor.add_connection(connected_system)
    
                # Add the Anchor object to the mooring map
                self.mooring_map.add_anchor(anchor)
                #print(f"Anchor added: {anchor.id}")


        # Now, iterate over the edges to add MooringLine instances
        for anchor_id, connected_system_id, edge_attrs in self.mooring_graph.edges(data=True):
            # Create a MooringLine instance using the edge attributes
            mooring_line = MooringLine(
                id=edge_attrs['mooringline_id'],
                anchor_id=anchor_id,
                mooringline_type=edge_attrs['type'],
                connected_to=connected_system_id
            )
            # Add the MooringLine to the mooring map
            self.mooring_map.add_mooring_line(mooring_line)

           # print(f"Mooring line added: {edge_attrs['mooringline_id']} from {anchor_id} to {connected_system_id}")

    
            # If the connected system is a turbine or substation, add reverse mapping to graph nodes
            if self._is_turbine(connected_system_id) or self._is_substation(connected_system_id):
                connected_systems_list = self.graph.nodes[connected_system_id].setdefault('mooring_systems', [])
                if anchor_id not in connected_systems_list:
                    connected_systems_list.append(anchor_id)

                   # print(f"Reverse mapping added for {connected_system_id}: {connected_systems_list}")

            
    # Log successful setup
        print(f"Mooring map created with {len(self.mooring_map.anchors)} anchors and {len(self.mooring_map.mooring_lines)} mooring lines.")
        

   
    def integrate_mooring_with_main_graph(self):
        # Assuming that `self.mooring_graph` and `self.graph` are already populated

        # Add all mooring nodes to the main graph
        for node_id, node_attrs in self.mooring_graph.nodes(data=True):
            self.graph.add_node(node_id, **node_attrs)

        # Add all mooring edges to the main graph
        for u, v, edge_attrs in self.mooring_graph.edges(data=True):
            # For an undirected mooring graph, add edges in both directions
            self.graph.add_edge(u, v, **edge_attrs)
            self.graph.add_edge(v, u, **edge_attrs)


    def create_combo_farm(self):
        """Integrates all components into a unified graph, ensuring mooring lines and anchors are included."""
        self.combo_graph = nx.MultiDiGraph()
        
        # Add nodes and edges from the main windfarm graph
        for node, data in self.graph.nodes(data=True):
            if 'system' in data:
                self.combo_graph.add_node(node, **data)
        for u, v, data in self.graph.edges(data=True):
            self.combo_graph.add_edge(u, v, **data)
        
        # Integrate mooring components (anchors and mooring lines)
        for anchor_id, data in self.mooring_graph.nodes(data=True):
            # Ensure the Mooring object is correctly accessed from node data
            mooring_obj = data.get('object')  # This should be your Mooring object
            if mooring_obj and hasattr(mooring_obj, 'connected_systems'):
                self.combo_graph.add_node(anchor_id, object=mooring_obj, type='mooring')
                # Add connections from the mooring to connected systems
                for system_id in mooring_obj.connected_systems:
                    if system_id in self.combo_graph and not self.combo_graph.has_edge(anchor_id, system_id):
                        self.combo_graph.add_edge(anchor_id, system_id, object=mooring_obj)
                    if system_id in self.combo_graph and not self.combo_graph.has_edge(system_id, anchor_id):
                        self.combo_graph.add_edge(system_id, anchor_id, object=mooring_obj)

                    

    def finish_setup(self) -> None:
        """Final initialization hook for any substations, turbines, or cables."""
        for start_node, end_node in self.graph.edges():
            if not (self.is_anchor(start_node) or self.is_anchor(end_node)):
                self.cable((start_node, end_node)).finish_setup()
            else:
                # Log or handle cases where a cable is attempted to be setup with an anchor
                #print(f"Skipping finish setup for cable between {start_node} and {end_node} as it involves an anchor.")
                continue


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
        System | Anchor
        The `System` object if it's a turbine or substation, or the `Anchor` object if it's an anchor.
        """
        node_data = self.graph.nodes[system_id]
        if 'system' in node_data:
            return node_data['system']
        elif 'object' in node_data and isinstance(node_data['object'], Mooring):
            # Assuming anchors are stored as Mooring objects in the 'object' field
            return node_data['object']
        else:
            raise KeyError(f"No system found with ID {system_id}.")


    @cache
    def combo_system(self, system_id: str) -> System:
        """Returns the desired `System` object or `Mooring` object from the combo_graph."""
        try:
            # Attempt to return a system object
            return self.combo_graph.nodes[system_id]['system']
        except KeyError:
            # Handle mooring objects differently if 'system' is not found
            if 'object' in self.combo_graph.nodes[system_id]:
                # Return the mooring object or a suitable representation if it's recognized as a mooring type
                if self.combo_graph.nodes[system_id]['type'] == 'mooring':
                    return self.combo_graph.nodes[system_id]['object']
            # Raise an error if neither a system nor a mooring object could be found
            raise KeyError(f"No system found with ID {system_id} in combo_graph.")

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
    

    @cache
    def fetch_mooring_component(self, id):
        """
        Helper method to fetch a mooring component based on its identifier.
        This can fetch either a node or an edge from the mooring graph.
        """
        if id in self.mooring_graph.nodes:
            return self.mooring_graph.nodes[id].get('object')  # Fetching node object
        for (u, v, data) in self.mooring_graph.edges(data=True):
            if data.get('mooringline_id') == id:
                return data  # Return edge data including mooring line details
        return None  # No valid component found

    
    @cache
    def mooring(self, identifier: str):
        """
        Convenience function to return a mooring component (either an anchor node or a mooring line edge) 
        in the windfarm based on its identifier.
    
        Parameters
        ----------
        identifier : str
            The unique identifier for either an anchor or a mooring line.
        
        Returns
        -------
        object
            The Mooring object if it's an anchor or details of the mooring line if it's a line.
        """
       
        """
        Returns a mooring component (either an anchor node or a mooring line edge) based on its identifier.
        """
        if ',' in identifier:  # Suggests multiple IDs in one string
            ids = identifier.split(',')
            results = []
            for id in ids:
                result = self.fetch_mooring_component(id.strip())
                if result:
                    results.append(result)
            return results if results else None
    
        # Single ID processing
        return self.fetch_mooring_component(identifier)
        

    


    @property
    def anchor_ids(self):
        """Extracts anchor IDs dynamically from the mooring_graph."""
        return [node for node, data in self.mooring_graph.nodes(data=True) if data.get('type') == 'anchor']
    
    @property
    def mooringline_ids(self):
        """Collects all mooring line IDs from the mooring_graph edges."""
        return [data['mooringline_id'] for _, _, data in self.mooring_graph.edges(data=True) if 'mooringline_id' in data]

    
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
