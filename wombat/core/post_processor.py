"""The postprocessing metric computation."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from pathlib import Path
from itertools import chain, product
from collections import Counter

import numpy as np
import pandas as pd

from wombat.core import Frequency, FixedCosts
from wombat.utilities import calculate_windfarm_operational_level
from wombat.core.library import load_yaml


def _check_frequency(frequency: str, which: str = "all") -> Frequency:
    """Checks the frequency input to ensure it meets the correct criteria according
    to the ``which`` flag.

    Parameters
    ----------
    frequency : str
        The user-provided value.
    which : str, optional
        Designation for which combinations to check for, by default "all".
        - "all": project, annual, monthly, and month-year

    Returns
    -------
    Frequency
        A :py:obj:`StrEnum` for the lower-case, input with white spaces removed.

    Raises
    ------
    ValueError
        Raised if an invalid value was raised
    """
    frequency = Frequency(frequency)
    if frequency not in (options := Frequency.options(which)):
        raise ValueError(f"`frequency` must be one of {options}.")
    return frequency


class Metrics:
    """The metric computation class for storing logs and compiling results."""

    _hourly_cost = "hourly_labor_cost"
    _salary_cost = "salary_labor_cost"
    _labor_cost = "total_labor_cost"
    _equipment_cost = "equipment_cost"
    _materials_cost = "materials_cost"
    _total_cost = "total_cost"
    _cost_columns = [
        _hourly_cost,
        _salary_cost,
        _labor_cost,
        _equipment_cost,
        _materials_cost,
        _total_cost,
    ]

    def __init__(
        self,
        data_dir: str | Path,
        events: str | pd.DataFrame,
        operations: str | pd.DataFrame,
        potential: str | pd.DataFrame,
        production: str | pd.DataFrame,
        inflation_rate: float,
        project_capacity: float,
        turbine_capacities: list[float],
        electrolyzer_rated_production: list[float],
        substation_id: str | list[str],
        turbine_id: str | list[str],
        electrolyzer_id: str | list[str],
        substation_turbine_map: dict[str, dict[str, list[str]]],
        service_equipment_names: str | list[str],
        fixed_costs: str | None = None,
    ) -> None:
        """Initializes the Metrics class.

        Parameters
        ----------
        data_dir : str | Path
            This should be the same as was used for running the analysis.
        events : str | pd.DataFrame
            Either a pandas ``DataFrame`` or filename to be used to read the .pqt log
            data.
        operations : str | pd.DataFrame
            Either a pandas ``DataFrame`` or filename to be used to read the .pqt log
            data.
        potential : str | pd.DataFrame
            Either a pandas ``DataFrame`` or a filename to be used to read the .pqt
            potential power production data.
        production : str | pd.DataFrame
            Either a pandas ``DataFrame`` or a filename to be used to read the .pqt
            power production data.
        inflation_rate : float
            The inflation rate to be applied to all dollar amounts from the analysis
            starting year to ending year.
        project_capacity : float
            The project's rated capacity, in MW.
        turbine_capacities : Union[float, List[float]]
            The capacity of each individual turbine corresponding to
            :py:attr`turbine_id`, in kW/hr.
        electrolyzer_rated_production : Union[float, List[float]]
            The rated production capacity of each individual electrolyzer corresponding
            to :py:attr:`electrolyzer_id`, in kg/hr.
        substation_id : str | list[str]
            The substation id(s).
        turbine_id : str | list[str]
            The turbine id(s).
        electrolyzer_id : str | list[str]
            The electrolyzer id(s).
        substation_turbine_map : dict[str, dict[str, list[str]]]
            A copy of ``Windfarm.substation_turbine_map``. This is a dictionary mapping
            of the subation IDs (keys) and a nested dictionary of its associated turbine
            IDs and each turbine's total plant weighting (turbine capacity / plant
            capacity).
        service_equipment_names : str | list[str]
            The names of the servicing equipment, corresponding to
            ``ServiceEquipment.settings.name`` for each ``ServiceEquipment`` in the
            simulation.
        fixed_costs : str | None
            The filename of the project's fixed costs.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"{self.data_dir} does not exist")

        self.inflation_rate = 1 + inflation_rate
        self.project_capacity = project_capacity

        if fixed_costs is None:
            # Create a zero-cost FixedCosts object
            self.fixed_costs = FixedCosts.from_dict({"operations": 0})
        else:
            if TYPE_CHECKING:
                assert isinstance(fixed_costs, str)
            fixed_costs = load_yaml(self.data_dir / "project/config", fixed_costs)
            if TYPE_CHECKING:
                assert isinstance(fixed_costs, dict)
            self.fixed_costs = FixedCosts.from_dict(fixed_costs)

        if isinstance(substation_id, str):
            substation_id = [substation_id]
        self.substation_id = substation_id

        if isinstance(turbine_id, str):
            turbine_id = [turbine_id]
        self.turbine_id = turbine_id

        if isinstance(electrolyzer_id, str):
            electrolyzer_id = [electrolyzer_id]
        self.electrolyzer_id = electrolyzer_id

        self.substation_turbine_map = substation_turbine_map
        self.turbine_weights = (
            pd.concat([pd.DataFrame(val) for val in substation_turbine_map.values()])
            .set_index("turbines")
            .T
        )

        if isinstance(service_equipment_names, str):
            service_equipment_names = [service_equipment_names]
        self.service_equipment_names = sorted(set(service_equipment_names))

        if isinstance(turbine_capacities, (float, int)):
            turbine_capacities = [turbine_capacities]
        self.turbine_capacities = np.array(turbine_capacities, dtype=float)

        if isinstance(electrolyzer_rated_production, (float, int)):
            electrolyzer_rated_production = [electrolyzer_rated_production]
        self.electrolyzer_rated_production = np.array(
            electrolyzer_rated_production, dtype=float
        )

        if isinstance(events, str):
            events = self._read_data(events)
        self.events = self._apply_inflation_rate(self._tidy_data(events))

        if isinstance(operations, str):
            operations = self._read_data(operations)
        self.operations = self._tidy_data(operations)
        self.operations["windfarm"] = calculate_windfarm_operational_level(
            operations=self.operations,
            turbine_id=self.turbine_id,
            turbine_weights=self.turbine_weights,
            substation_turbine_map=self.substation_turbine_map,
        )

        if isinstance(potential, str):
            potential = self._read_data(potential)
        self.potential = self._tidy_data(potential)

        if isinstance(production, str):
            production = self._read_data(production)
        self.production = self._tidy_data(production)

        prod_cols = self.turbine_id + self.electrolyzer_id
        self.potential[prod_cols] = self.potential[prod_cols].astype(float)
        self.production[prod_cols] = self.production[prod_cols].astype(float)

    def __eq__(self, other) -> bool:
        """Check that the essential information is the same."""
        if isinstance(other, Metrics):
            checks = []
            for _, expected, actual in self._yield_comparisons(other):
                match expected:
                    case pd.DataFrame() | pd.Series():
                        checks.append(expected.equals(actual))
                    case np.ndarray():
                        checks.append(np.array_equal(expected, actual))
                    case _:
                        checks.append(expected == actual)
            return all(checks)
        return False

    def __ne__(self, other) -> bool:
        """Checks for object inequality."""
        return not (self == other)

    def _yield_comparisons(self, other):
        """Returns the name and individual class attributes to compare for equality
        tests.
        """
        to_compare = [
            "data_dir",
            "events",
            "operations",
            "inflation_rate",
            "project_capacity",
            "potential",
            "production",
            "service_equipment_names",
            "turbine_id",
            "electrolyzer_id",
            "substation_id",
            "fixed_costs",
            "turbine_capacities",
            "electrolyzer_rated_production",
            "substation_turbine_map",
            "turbine_weights",
        ]
        dataframes = [
            "operations",
            "events",
            "potential",
            "production",
            "turbine_weights",
        ]
        for name in to_compare:
            if name in dataframes:
                yield (
                    name,
                    getattr(self, name).sort_index(axis=1),
                    getattr(other, name).sort_index(axis=1),
                )
            else:
                yield name, getattr(self, name), getattr(other, name)

    def _repr_compare(self, other) -> list[str]:
        """Comparison representation."""
        explanation = []
        for name, expected, actual in self._yield_comparisons(other):
            if isinstance(expected, pd.DataFrame):
                if expected.equals(actual):
                    continue
            elif expected == actual:
                continue
            if isinstance(expected, pd.DataFrame):
                if isinstance(actual, pd.DataFrame):
                    explanation.extend(
                        [
                            f"`{name}` is not equal:",
                            f"Expected shape {expected.shape}, got: {actual.shape}",
                            f"Expected columns: {expected.columns}, got: {actual.columns}",  # noqa: E501
                            f"Expected dtypes: {expected.dtypes}, got: {actual.dtypes}",
                        ]
                    )
                else:
                    explanation.append(
                        f"Expected a dataframe for `{name}`,"
                        f" received object of type: {type(actual)}"
                    )
            else:
                explanation.extend(
                    [
                        f"Comparing `{name}`",
                        f"expected: {expected}",
                        f"     got: {actual}",
                    ]
                )
        return explanation

    @classmethod
    def from_simulation_outputs(cls, fpath: Path | str, fname: str) -> Metrics:
        """Creates the Metrics class from the saved outputs of a simulation for ease of
        revisiting the calculated metrics.

        Parameters
        ----------
        fpath : Path | str
            The full path to the file where the data was saved.
        fname : Path | str
            The filename for where the data was saved, which should be a direct
            dictionary mapping for the Metrics initialization.

        Returns
        -------
        Metrics
            The class object.
        """
        data = load_yaml(fpath, fname)
        metrics = cls(**data)
        return metrics

    def _tidy_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Tidies the "raw" parquet-converted data to be able to be used among the
        ``Metrics`` class.

        Parameters
        ----------
        data : pd.DataFrame
            The tabular log data.

        Returns
        -------
        pd.DataFrame
            A tidied data frame to be used for all the operations in this class.
        """
        # Ignore odd pandas casting error for pandas>=1.5(?)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = data = data.convert_dtypes()

        if data.index.name != "datetime":
            try:
                data.datetime = pd.to_datetime(data.datetime)
            except AttributeError:
                data["datetime"] = pd.to_datetime(data.env_datetime)
            data.index = data.datetime
            data = data.drop(labels="datetime", axis=1)
        data.env_datetime = pd.to_datetime(data.env_datetime)
        data = data.assign(
            year=data.env_datetime.dt.year,
            month=data.env_datetime.dt.month,
            day=data.env_datetime.dt.day,
        )
        return data

    def _read_data(self, fname: str) -> pd.DataFrame:
        """Reads the Parquet log data from library's results folder.

        Parameters
        ----------
        fname : str
            Filename of the parquet data.

        Returns
        -------
        pd.DataFrame
            Dataframe of either the events or operations data.
        """
        return pd.read_parquet(self.data_dir / "results" / fname)

    def _apply_inflation_rate(self, events: pd.DataFrame) -> pd.DataFrame:
        """Adjusts the cost data for compounding inflation.

        Parameters
        ----------
        inflation_rate : float
            The inflation rate to be applied for each year.
        events : pd.DataFrame
            The events dataframe containing the project cost data.

        Returns
        -------
        pd.DataFrame
            The events dataframe with costs adjusted for inflation.
        """
        adjusted_inflation = deepcopy(self.inflation_rate)
        years = events.year.unique()
        years.sort()
        for year in years:
            row_filter = events.year == year
            if year > years[0]:
                events.loc[row_filter, self._cost_columns] *= adjusted_inflation
                adjusted_inflation *= self.inflation_rate

        return events

    def time_based_availability(self, frequency: str, by: str) -> pd.DataFrame:
        """Calculates the time-based availabiliy over a project's lifetime for the
        wind farm, turbine(s) or electrolyzer(s). Time-based availability is the
        proportion of total uptime, regardless of operational capacity.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by : str
            One of "windfarm", "turbine", or "electrolyzer". Electrolyzer results are
            not incorporated into the overall wind farm availability levels. As such,
            only electrolyzer outputs are provided for the electrolyzer.

        Returns
        -------
        pd.DataFrame
            The time-based availability at the desired aggregation level.

        Raises
        ------
        ValueError
            Raised when :py:attr:`by` == "electrolyzer" and there were no simulated
            electrolyzers.
        """
        frequency = _check_frequency(frequency, which="all")

        by = by.lower().strip()
        if by not in ("windfarm", "turbine", "electrolyzer"):
            raise ValueError(
                '`by` must be one of "windfarm", "turbine", or "electrolyzer".'
            )

        by_windfarm = by == "windfarm"
        by_electrolyzer = by == "electrolyzer"

        if by_electrolyzer and self.electrolyzer_rated_production.size == 0:
            raise ValueError("No electrolyzers available to analyze.")

        time_cols = frequency.group_cols

        if by_electrolyzer:
            _id = self.electrolyzer_id
        else:
            _id = self.turbine_id

        operations_cols = time_cols + _id
        operations = self.operations.loc[:, operations_cols]
        operations.loc[:, _id] = operations[_id] > 0

        if frequency is Frequency.PROJECT:
            if by_windfarm:
                availability = pd.DataFrame(
                    [operations.values.sum() / operations.size],
                    columns=["windfarm"],
                    index=["time_availability"],
                )
                return availability
            availability = (
                operations.sum(axis=0).to_frame("time_availability").T
                / operations.shape[0]
            )
            return availability

        if by_windfarm:
            availability = operations.groupby(time_cols).sum().sum(axis=1).to_frame(
                "windfarm"
            ) / operations.groupby(time_cols).count().sum(axis=1).to_frame("windfarm")
            return availability
        availability = (
            operations.groupby(time_cols).sum() / operations.groupby(time_cols).count()
        )
        return availability

    def production_based_availability(self, frequency: str, by: str) -> pd.DataFrame:
        """Calculates the production-based availabiliy over a project's lifetime for the
        wind farm, turbine(s) or electrolyzer(s). Production-based availability is the
        produced energy divided by the potential energy.

        .. note:: There is not currently a power curve model for electrolyzers, so the
            power potential at each time step is 100%, and the power production is the
            operational capacity.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by : str
            One of "windfarm", "turbine", or "electrolyzer".

        Returns
        -------
        pd.DataFrame
            The production-based availability at the desired aggregation level.

        Raises
        ------
        ValueError
            Raised when :py:attr:`by` == "electrolyzer" and there were no simulated
            electrolyzers.
        """
        frequency = _check_frequency(frequency, which="all")

        by = by.lower().strip()
        if by not in ("windfarm", "turbine", "electrolyzer"):
            raise ValueError(
                '`by` must be one of "windfarm", "turbine", or "electrolyzer".'
            )

        by_windfarm = by == "windfarm"
        by_electrolyzer = by == "electrolyzer"

        if by_electrolyzer and self.electrolyzer_rated_production.size == 0:
            raise ValueError("No electrolyzers available to analyze.")

        time_cols = frequency.group_cols

        if by_electrolyzer:
            operations_cols = time_cols + self.electrolyzer_id
            production = self.operations.loc[:, operations_cols]
            potential = production.copy()
            potential.loc[:, self.electrolyzer_id] = 1.0
        else:
            operations_cols = time_cols + self.turbine_id
            production = self.production.loc[:, operations_cols]
            potential = self.potential.loc[:, operations_cols]

        if frequency is Frequency.PROJECT:
            if by_windfarm:
                production = production.values.sum()
                potential = potential.values.sum()
                potential = 1 if potential == 0 else potential
                availability = pd.DataFrame(
                    [production / potential],
                    columns=["windfarm"],
                    index=["energy_availability"],
                )
                return availability

            production = production.sum(axis=0).to_frame("energy_availability").T
            potential = (
                potential.sum(axis=0).to_frame("energy_availability").T.replace(0, 1)
            )
            return production / potential

        if by_windfarm:
            potential = (
                potential.groupby(time_cols)
                .sum()
                .sum(axis=1)
                .to_frame("windfarm")
                .replace(0, 1)
            )
            production = (
                production.groupby(time_cols).sum().sum(axis=1).to_frame("windfarm")
            )
            return production / potential

        production = production.groupby(time_cols).sum()
        potential = potential.groupby(time_cols).sum().replace(0, 1)
        return production / potential

    def capacity_factor(self, which: str, frequency: str, by: str) -> pd.DataFrame:
        """Calculates the capacity factor over a project's lifetime as a single value,
        annual average, or monthly average for the whole windfarm or by turbine.

        .. note:: This currently assumes that if there are multiple substations, that
          the turbines are all connected to multiple.


        Parameters
        ----------
        which : str
            One of "net" or "gross".
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by : str
            One of "windfarm", "turbine", or "electrolyzer".

        Returns
        -------
        pd.DataFrame
            The capacity factor at the desired aggregation level.

        Raises
        ------
        ValueError
            Raised when :py:attr:`by` == "electrolyzer" and there were no simulated
            electrolyzers.
        """
        which = which.lower().strip()
        if which not in ("net", "gross"):
            raise ValueError('``which`` must be one of "net" or "gross".')

        frequency = _check_frequency(frequency, which="all")
        time_cols = frequency.group_cols

        by = by.lower().strip()
        if by not in ("windfarm", "turbine", "electrolyzer"):
            raise ValueError(
                '`by` must be one of "windfarm", "turbine", or "electrolyzer".'
            )

        by_windfarm = by == "windfarm"
        by_electrolyzer = by == "electrolyzer"

        if by_electrolyzer and self.electrolyzer_rated_production.size == 0:
            raise ValueError("No electrolyzers available to analyze.")

        if by_electrolyzer:
            _id = self.electrolyzer_id
            capacities = self.electrolyzer_rated_production
        else:
            _id = self.turbine_id
            capacities = self.turbine_capacities

        cols = time_cols + _id
        production = self.production if which == "net" else self.potential
        production = production.loc[:, cols]
        capacity = production.copy()
        capacity.loc[:, _id] = np.full(production.loc[:, _id].shape, capacities)

        if frequency is Frequency.PROJECT:
            name = f"{which}_capacity_factor"
            if by_windfarm:
                cf = pd.DataFrame(
                    [production.values.sum() / capacity.values.sum()],
                    columns=["windfarm"],
                    index=[name],
                )
                return cf

            production = production.sum(axis=0).to_frame(name).T
            capacity = capacity.sum(axis=0).to_frame(name).T.replace(0, 1)
            return production / capacity

        if by_windfarm:
            production = (
                production.groupby(time_cols)
                .sum()
                .sum(axis=1)
                .to_frame("windfarm")
                .replace(0, 1)
            )
            capacity = (
                capacity.groupby(time_cols).sum().sum(axis=1).to_frame("windfarm")
            )
            return production / capacity

        production = production.groupby(time_cols).sum()
        capacity = capacity.groupby(time_cols).sum().replace(0, 1)
        return production / capacity

    def task_completion_rate(self, which: str, frequency: str) -> float | pd.DataFrame:
        """Calculates the task completion rate (including tasks that are canceled after
        a replacement event) over a project's lifetime as a single value, annual
        average, or monthly average for the whole windfarm or by turbine.

        Parameters
        ----------
        which : str
            One of "scheduled", "unscheduled", or "both".
        frequency : str
            One of "project", "annual", "monthly", or "month-year".

        Returns
        -------
        float | pd.DataFrame
            The task completion rate at the desired aggregation level.
        """
        which = which.lower().strip()
        if which not in ("scheduled", "unscheduled", "both"):
            raise ValueError(
                '``which`` must be one of "scheduled", "unscheduled", or "both".'
            )

        frequency = _check_frequency(frequency, which="all")

        if which == "scheduled":
            task_filter = ["maintenance"]
        elif which == "unscheduled":
            task_filter = ["repair"]
        else:
            task_filter = ["maintenance", "repair"]

        cols = ["env_datetime", "request_id"]
        request_filter = [f"{el} request" for el in task_filter]
        completion_filter = [
            f"{task} {el}" for task in task_filter for el in ("complete", "canceled")
        ]
        requests = self.events.loc[
            self.events.action.isin(request_filter), cols
        ].reset_index(drop=True)
        completions = self.events.loc[
            self.events.action.isin(completion_filter), cols
        ].reset_index(drop=True)

        if frequency is Frequency.PROJECT:
            if requests.shape[0] == 0:
                return pd.DataFrame([0.0], columns=["windfarm"])
            return pd.DataFrame(
                [completions.shape[0] / requests.shape[0]], columns=["windfarm"]
            )

        requests["year"] = requests.env_datetime.dt.year.values
        requests["month"] = requests.env_datetime.dt.month.values

        completions["year"] = completions.env_datetime.dt.year.values
        completions["month"] = completions.env_datetime.dt.month.values

        group_filter = frequency.group_cols
        if frequency is Frequency.ANNUAL:
            indices = self.operations.year.unique()
        elif frequency is Frequency.MONTHLY:
            indices = self.operations.month.unique()
        elif frequency is Frequency.MONTH_YEAR:
            indices = (
                self.operations[["year", "month"]]
                .groupby(group_filter)
                .value_counts()
                .index.tolist()
            )

        group_cols = group_filter + ["request_id"]
        requests = requests[group_cols].groupby(group_filter).count()
        requests.loc[requests.request_id == 0] = 1
        completions = completions[group_cols].groupby(group_filter).count()

        missing = [ix for ix in indices if ix not in requests.index]
        requests = pd.concat(
            [
                requests,
                pd.DataFrame(
                    np.ones(len(missing)), index=missing, columns=requests.columns
                ),
            ]
        ).sort_index()

        missing = [ix for ix in indices if ix not in completions.index]
        completions = pd.concat(
            [
                completions,
                pd.DataFrame(
                    np.ones(len(missing)), index=missing, columns=completions.columns
                ),
            ]
        ).sort_index()

        completion_rate = pd.DataFrame(completions / requests)
        completion_rate.index = completion_rate.index.set_names(group_filter)
        return completion_rate.rename(
            columns={"request_id": "Completion Rate", 0: "Completion Rate"}
        )

    def equipment_costs(
        self, frequency: str, by_equipment: bool = False
    ) -> pd.DataFrame:
        """Calculates the equipment costs for the simulation at a project, annual, or
        monthly level with (or without) respect to equipment utilized in the simulation.
        This excludes any port fees that might apply, which are included in:
        ``port_fees``.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by_equipment : bool, optional
            Indicates whether the values are with resepect to the equipment utilized
            (True) or not (False), by default False.

        Returns
        -------
        pd.DataFrame
            Returns pandas ``DataFrame`` with columns:
                - year (if appropriate for frequency)
                - month (if appropriate for frequency)
                - then any equipment names as they appear in the logs

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or
            "month-year".
        ValueError
            If ``by_equipment`` is not one of ``True`` or ``False``.
        """
        frequency = _check_frequency(frequency, which="all")

        if not isinstance(by_equipment, bool):
            raise ValueError("`by_equipment` must be one of `True` or `False`")

        col_filter = frequency.group_cols

        cost_col = [self._equipment_cost]
        events = self.events.loc[self.events.action != "monthly lease fee"]
        if by_equipment:
            if frequency is Frequency.PROJECT:
                costs = (
                    events.loc[events[self._equipment_cost] > 0, cost_col + ["agent"]]
                    .groupby(["agent"])
                    .sum()
                    .fillna(0)
                    .reset_index(level=0)
                    .fillna(0)
                    .T
                )
                costs = (
                    costs.rename(columns=costs.iloc[0])
                    .drop(index="agent")
                    .reset_index(drop=True)
                )
                return costs

            col_filter = ["agent"] + col_filter
            costs = (
                events.loc[events[self._equipment_cost] > 0, cost_col + col_filter]
                .groupby(col_filter)
                .sum()
                .reset_index(level=0)
            )
            costs = pd.concat(
                [
                    costs[costs.agent == eq][cost_col].rename(
                        columns={self._equipment_cost: eq}
                    )
                    for eq in costs.agent.unique()
                ],
                axis=1,
            )
            return costs.fillna(value=0).sort_index()

        if frequency is Frequency.PROJECT:
            return pd.DataFrame([events[cost_col].sum()], columns=cost_col)

        costs = events[cost_col + col_filter].groupby(col_filter).sum()
        return costs.fillna(0).sort_index()

    def service_equipment_utilization(self, frequency: str) -> pd.DataFrame:
        """Calculates the utilization rate for each of the service equipment in the
        simulation  as the ratio of total number of days each of the servicing
        equipment is in operation over the total number of days it's present in the
        simulation. This number excludes mobilization time and the time between
        visits for scheduled servicing equipment strategies.

        .. note:: For tugboats in a tow-to-port scenario, this ratio will be near
            100% because they are considered to be operating on an as-needed basis per
            the port contracting assumptions

        Parameters
        ----------
        frequency : str
            One of "project" or "annual".

        Returns
        -------
        pd.DataFrame
            The utilization rate of each of the simulation ``SerivceEquipment``.

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project" or "annual".
        """
        frequency = _check_frequency(frequency, which="annual")

        operation_days = []
        total_days = []
        operating_actions = [
            "traveling",  # traveling between port/site or on-site
            "transferring crew",
            "repair",
            "maintenance",
            "delay",  # performing work
            "unmooring",
            "mooring_reconnection",
            "towing",  # tugboat classifications
        ]
        operating_filter = self.events.action.isin(operating_actions)
        return_filter = self.events.action == "delay"
        return_filter &= (
            (self.events.reason == "work is complete")
            & (self.events.additional == "will return next year")
        ) | (self.events.reason == "non-operational period")
        return_filter &= self.events.additional == "will return next year"
        for name in self.service_equipment_names:
            equipment_filter = self.events.agent == name
            _events = self.events[equipment_filter & operating_filter]
            _events = _events.groupby(["year", "month", "day"]).size()
            _events = _events.reset_index().groupby("year").count()[["day"]]
            operation_days.append(_events.rename(columns={"day": name}))

            ix_filter = equipment_filter & ~return_filter
            total = self.events[ix_filter].groupby(["year", "month", "day"]).size()
            total = total.reset_index().groupby("year").count()[["day"]]
            total_days.append(total.rename(columns={"day": name}))

        operating_df = pd.DataFrame(operation_days[0])
        total_df = pd.DataFrame(total_days[0])
        if len(self.service_equipment_names) > 1:
            operating_df = operating_df.join(operation_days[1:], how="outer").fillna(0)
            total_df = total_df.join(total_days[1:], how="outer").fillna(1)

        for year in self.events.year.unique():
            if year not in operating_df.index:
                missing = pd.DataFrame(
                    np.zeros((1, operating_df.shape[1])),
                    index=[year],
                    columns=operating_df.columns,
                )
                operating_df = pd.concat([operating_df, missing], axis=0).sort_index()
            if year not in total_df.index:
                missing = pd.DataFrame(
                    np.ones((1, total_df.shape[1])),
                    index=[year],
                    columns=operating_df.columns,
                )
                total_df = pd.concat([total_df, missing], axis=0).sort_index()

        if frequency is Frequency.PROJECT:
            operating_df = operating_df.reset_index().sum()[
                self.service_equipment_names
            ]
            total_df = total_df.reset_index().sum()[self.service_equipment_names]
            return pd.DataFrame(operating_df / total_df).T
        return operating_df / total_df

    def vessel_crew_hours_at_sea(
        self,
        frequency: str,
        by_equipment: bool = False,
        vessel_crew_assumption: dict[str, float] = {},
    ) -> pd.DataFrame:
        """Calculates the total number of crew hours at sea that occurred during a
        simulation at a project, annual, or monthly level that can be broken out by
        servicing equipment. This includes time mobilizing, delayed at sea, servicing,
        towing, and traveling.

        .. note:: This metric is intended to be used for offshore wind simulations.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by_equipment : bool, optional
            Indicates whether the values are with resepect to each tugboat (True) or not
            (False), by default False.
        vessel_crew_assumption : dict[str, float], optional
            Dictionary of vessel names (``ServiceEquipment.settings.name``) and number
            of crew members aboard to trannsform the results from vessel hours at sea
            to crew hours at sea.

        Returns
        -------
        pd.DataFrame
            Returns a pandas ``DataFrame`` with columns:

            - year (if appropriate for frequency)
            - month (if appropriate for frequency)
            - Total Crew Hours at Sea
            - {ServiceEquipment.settings.name} (if broken out)

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or
            "month-year".
        ValueError
            If ``by_equipment`` is not one of ``True`` or ``False``.
        ValueError
            If ``vessel_crew_assumption`` is not a dictionary.
        """
        frequency = _check_frequency(frequency, which="all")

        if not isinstance(by_equipment, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")

        if not isinstance(vessel_crew_assumption, dict):
            raise ValueError(
                "`vessel_crew_assumption` must be a dictionary of vessel name (keys)"
                " and number of crew (values)"
            )

        # Filter by the at sea indicators and required columns
        at_sea = self.events
        at_sea = at_sea.loc[
            at_sea.location.isin(("enroute", "site", "system"))
            & at_sea.agent.isin(self.service_equipment_names),
            ["agent", "year", "month", "action", "reason", "additional", "duration"],
        ].reset_index(drop=True)

        # Create a shell for the final results
        total_hours = (
            self.events[["env_time", "year", "month"]]
            .groupby(["year", "month"])
            .count()
        )
        total_hours = total_hours.reset_index().rename(columns={"env_time": "N"})
        total_hours.N = 0

        # Apply the vessel crew assumptions
        vessels = at_sea.agent.unique()
        if vessel_crew_assumption != {}:
            for name, n_crew in vessel_crew_assumption.items():
                if name not in vessels:
                    continue
                ix_vessel = at_sea.agent == name
                at_sea.loc[ix_vessel, "duration"] *= n_crew

        group_cols = ["agent"]
        columns = ["Total Crew Hours at Sea"] + vessels.tolist()
        if not by_equipment:
            group_cols.pop(0)
            columns = ["Total Crew Hours at Sea"]
            at_sea = at_sea.groupby(["year", "month"]).sum()[["duration"]].reset_index()

        if frequency is Frequency.PROJECT:
            total_hours = pd.DataFrame([[0]], columns=["duration"])
            if by_equipment:
                total_hours = (
                    at_sea[["duration", "agent"]]
                    .groupby(["agent"])
                    .sum()
                    .T.reset_index(drop=True)
                )
                total_hours.loc[:, "Total Crew Hours at Sea"] = total_hours.sum().sum()
                return total_hours[columns]
            else:
                return pd.DataFrame(at_sea.sum()[["duration"]]).T.rename(
                    columns={"duration": "Total Crew Hours at Sea"}
                )
        additional_cols = frequency.group_cols
        total_hours = (
            total_hours.drop(columns=frequency.drop_cols)
            .groupby(group_cols)[["N"]]
            .sum()
        )

        columns = additional_cols + columns
        group_cols.extend(additional_cols)
        at_sea = at_sea[group_cols + ["duration"]].groupby(group_cols).sum()
        if by_equipment:
            total = []
            for v in vessels:
                total.append(at_sea.loc[v].rename(columns={"duration": v}))
            total_hours = total_hours.join(
                pd.concat(total, axis=1), how="outer"
            ).fillna(0)
            total_hours.N = total_hours.sum(axis=1)
            total_hours = (
                total_hours.reset_index()
                .rename(columns={"N": "Total Crew Hours at Sea"})[columns]
                .set_index(additional_cols)
            )
            return total_hours

        return at_sea.rename(columns={"duration": "Total Crew Hours at Sea"})

    def number_of_tows(
        self, frequency: str, by_tug: bool = False, by_direction: bool = False
    ) -> float | pd.DataFrame:
        """Calculates the total number of tows that occurred during a simulation at a
        project, annual, or monthly level that can be broken out by tugboat.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by_tug : bool, optional
            Indicates whether the values are with resepect to each tugboat (True) or not
            (False), by default False.
        by_direction : bool, optional
            Indicates whether the values are with respect to the direction a turbine is
            towed (True) or not (False), by default False.

        Returns
        -------
        float | pd.DataFrame
            Returns either a float for whole project-level costs or a pandas
            ``DataFrame`` with columns:

            - year (if appropriate for frequency)
            - month (if appropriate for frequency)
            - total_tows
            - total_tows_to_port (if broken out)
            - total_tows_to_site (if broken out)
            - {ServiceEquipment.settings.name}_total_tows (if broken out)
            - {ServiceEquipment.settings.name}_to_port (if broken out)
            - {ServiceEquipment.settings.name}_to_site (if broken out)

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or
            "month-year".
        ValueError
            If ``by_tug`` is not one of ``True`` or ``False``.
        ValueError
            If ``by_direction`` is not one of ``True`` or ``False``.
        """
        frequency = _check_frequency(frequency, which="all")

        if not isinstance(by_tug, bool):
            raise ValueError("``by_tug`` must be one of ``True`` or ``False``")

        if not isinstance(by_direction, bool):
            raise ValueError("``by_direction`` must be one of ``True`` or ``False``")

        # Filter out only the towing events
        towing = self.events.loc[self.events.action == "towing"].copy()
        if towing.shape[0] == 0:
            # If this is accessed in an in-situ only scenario, or no tows were activated
            # then return back 0
            return pd.DataFrame([[0]], columns=["total_tows"])
        towing.loc[:, "direction"] = "to_site"
        ix_to_port = towing.reason == "towing turbine to port"
        towing.loc[ix_to_port, "direction"] = "to_port"

        # Get the unique directions and tugboat names
        direction_suffix = ("to_port", "to_site")
        tugboats = towing.agent.unique().tolist()

        # Create the final column names
        columns = ["total_tows"]
        if by_direction:
            columns.extend([f"{c}_{s}" for c in columns for s in direction_suffix])

        if by_tug:
            tug_columns = [f"{t}_total_tows" for t in tugboats]
            if by_direction:
                _columns = [f"{t}_{s}" for t in tugboats for s in direction_suffix]
                tug_columns.extend(_columns)
                tug_columns.sort()
            columns.extend(tug_columns)

        # Count the total number of tows by each possibly category
        time_cols = frequency.group_cols
        group_cols = deepcopy(time_cols)
        if by_tug:
            group_cols.append("agent")
        if by_direction:
            group_cols.append("direction")
        if not group_cols:
            group_cols = ["agent"]
        n_tows = (
            towing[[*group_cols, "env_time"]]
            .groupby(group_cols)
            .count()
            .rename(columns={"env_time": "N"})
        )

        # Create the correct time frequency for the number of tows and shell total
        if frequency is Frequency.PROJECT:
            # If no further work is required, then return the sum as a 1x1 data frame
            if not by_tug and not by_direction:
                return pd.DataFrame(
                    [n_tows.reset_index().N.sum()], columns=["total_tows"]
                )

        if not by_tug and not by_direction:
            return n_tows.rename(columns={"N": "total_tows"})

        if by_tug and not by_direction:
            if frequency is Frequency.PROJECT:
                n_tows = (
                    n_tows.assign(total="total")
                    .set_index("total", append=True)
                    .swaplevel("agent", "total")
                )
            total_tows = n_tows.unstack().droplevel(0, axis=1).fillna(0)
            total_tows["total_tows"] = total_tows.sum(axis=1)
            total_tows = total_tows.rename(
                columns={t: f"{t}_total_tows" for t in tugboats}
            )[columns]
            total_tows.columns.name = None
            return total_tows

        if not by_tug and by_direction:
            if frequency is Frequency.PROJECT:
                n_tows = (
                    n_tows.assign(total="total")
                    .set_index("total", append=True)
                    .swaplevel("direction", "total")
                )
            total_tows = n_tows.unstack().droplevel(0, axis=1).fillna(0)
            total_tows["total_tows"] = total_tows.sum(axis=1)
            total_tows = total_tows.rename(
                columns={s: f"total_tows_{s}" for s in direction_suffix}
            )[columns]
            total_tows.columns.name = None
            return total_tows

        if frequency is Frequency.PROJECT:
            n_tows = (
                n_tows.assign(total="total")
                .set_index("total", append=True)
                .swaplevel("direction", "total")
                .swaplevel("agent", "total")
            )
            time_cols = ["total"]
        tug_sums = (
            n_tows.swaplevel("agent", "direction")
            .unstack()
            .droplevel(0, axis=1)
            .fillna(0)
        )
        tug_sums["total_tows"] = tug_sums.sum(axis=1)
        tug_sums = (
            tug_sums.rename(columns={t: f"{t}_total_tows" for t in tugboats})
            .droplevel("direction")
            .reset_index(drop=False)
            .groupby(time_cols)
            .sum()
        )

        dir_sums = (
            n_tows.unstack()
            .droplevel(0, axis=1)
            .fillna(0)
            .rename(columns={s: f"total_tows_{s}" for s in direction_suffix})
            .droplevel("agent")
            .reset_index(drop=False)
            .groupby(time_cols)
            .sum()
        )
        tug_dir_sums = n_tows.unstack().droplevel(0, axis=1).fillna(0).unstack()
        tugs = tug_dir_sums.columns.get_level_values("agent")
        dirs = tug_dir_sums.columns.get_level_values("direction")
        cols = ["_".join(el) for el in zip(tugs, dirs)]
        tug_dir_sums.columns = cols

        return pd.concat([tug_sums, dir_sums, tug_dir_sums], axis=1)[columns]

    def dispatch_summary(self, frequency: str) -> pd.DataFrame:
        """Calculates the total number of mobilizations for each servicing equipment
        and the average number of charter days for each dispatching.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".

        Returns
        -------
        pd.DataFrame
            Data frame of each servicing equipment and its number of mobilizations and
            their average chartering length.
        """
        frequency = _check_frequency(frequency, which="all")

        ev = self.events
        group_cols = [*frequency.group_cols, "agent"]
        average_charter_days = []
        for name in self.service_equipment_names:
            mobilizations = (
                ev.loc[
                    (
                        ev.action.eq("mobilization")
                        & ev.reason.str.contains("arrived on site")
                    )
                    & ev.agent.eq(name),
                    [*group_cols, "env_time"],
                ]
                .reset_index(drop=True)
                .rename(columns={"env_time": "mobilized"})
            )
            leaving = (
                ev.loc[ev.action.eq("leaving site") & ev.agent.eq(name), ["env_time"]]
                .reset_index(drop=True)
                .rename(columns={"env_time": "leaving"})
            )
            if leaving.shape[0] == 0:
                leaving = (
                    ev.loc[ev.agent.eq(name), ["env_time"]]
                    .tail(1)
                    .reset_index(drop=True)
                    .rename(columns={"env_time": "leaving"})
                )
            if mobilizations.shape[0] - leaving.shape[0] == 1:
                mobilizations = mobilizations.iloc[:-1]
            charter_days = pd.concat(
                [mobilizations.reset_index(drop=True), leaving.reset_index(drop=True)],
                axis=1,
            )
            charter_days = (
                charter_days.assign(
                    charter_days=(charter_days.leaving - charter_days.mobilized) / 24
                )
                .drop(columns=["mobilized", "leaving"])
                .groupby(group_cols)
                .agg(["count", "mean"])
                .droplevel(0, axis=1)
                .rename(
                    columns={"count": "N Mobilizations", "mean": "Average Charter Days"}
                )
            )
            average_charter_days.append(charter_days)
        average_charter_days = pd.concat(average_charter_days).sort_index()

        return average_charter_days

    def labor_costs(
        self, frequency: str, by_type: bool = False
    ) -> float | pd.DataFrame:
        """Calculates the labor costs for the simulation at a project, annual, or
        monthly level that can be broken out by hourly and salary labor costs.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by_type : bool, optional
            Indicates whether the values are with resepect to the labor types
            (True) or not (False), by default False.

        Returns
        -------
        float | pd.DataFrame
            Returns either a float for whole project-level costs or a pandas
            ``DataFrame`` with columns:

            - year (if appropriate for frequency)
            - month (if appropriate for frequency)
            - total_labor_cost
            - hourly_labor_cost (if broken out)
            - salary_labor_cost (if broken out)

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or
            "month-year".
        ValueError
            If ``by_type`` is not one of ``True`` or ``False``.
        """
        frequency = _check_frequency(frequency, which="all")

        if not isinstance(by_type, bool):
            raise ValueError("``by_type`` must be one of ``True`` or ``False``")

        labor_cols = [self._hourly_cost, self._salary_cost, self._labor_cost]
        if frequency is Frequency.PROJECT:
            costs = pd.DataFrame(
                self.events[labor_cols].sum(axis=0).values.reshape(1, -1),
                columns=labor_cols,
            )
            if not by_type:
                return costs[[self._labor_cost]]
            return costs

        group_filter = frequency.group_cols

        costs = (
            self.events.loc[:, labor_cols + group_filter]
            .groupby(group_filter)
            .sum()
            .fillna(value=0)
        )
        if not by_type:
            return pd.DataFrame(costs[self._labor_cost])
        return costs

    def equipment_labor_cost_breakdowns(
        self,
        frequency: str,
        by_category: bool = False,
        by_equipment: bool = False,
    ) -> pd.DataFrame:
        """Calculates the producitivty cost and time breakdowns for the simulation at a
        project, annual, or monthly level that can be broken out to include the
        equipment and labor components, as well as be broken down by servicing
        equipment.

        .. note:: Doesn't produce a value if there's no cost associated with a "reason".

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by_category : bool, optional
            Indicates whether to include the equipment and labor categories (True) or
            not (False), by default False.
        by_equipment : bool, optional
            Indicates whether the values are with resepect to the equipment utilized
            (True) or not (False), by default False.

        Returns
        -------
        pd.DataFrame
            Returns pandas ``DataFrame`` with columns:
                - year (if appropriate for frequency)
                - month (if appropriate for frequency)
                - reason
                - hourly_labor_cost (if by_category == ``True``)
                - salary_labor_cost (if by_category == ``True``)
                - total_labor_cost (if by_category == ``True``)
                - equipment_cost (if by_category == ``True``)
                - total_cost (if broken out)
                - total_hours

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or
            "month-year".
        ValueError
            If ``by_category`` is not one of ``True`` or ``False``.
        """
        frequency = _check_frequency(frequency, which="all")
        if not isinstance(by_category, bool):
            raise ValueError("``by_category`` must be one of ``True`` or ``False``")
        if not isinstance(by_equipment, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")

        group_filter = frequency.group_cols
        if by_equipment:
            group_filter.append("agent")
        group_filter.extend(["action", "reason", "additional"])

        action_list = [
            "delay",
            "repair",
            "maintenance",
            "mobilization",
            "transferring crew",
            "traveling",
            "towing",
            "unmooring",
            "mooring reconnection",
        ]
        equipment = self.events[self.events[self._equipment_cost] > 0].agent.unique()
        costs = (
            self.events.loc[
                self.events.agent.isin(equipment)
                & self.events.action.isin(action_list)
                & ~self.events.additional.isin(["work is complete"]),
                group_filter + self._cost_columns + ["duration"],
            ]
            .groupby(group_filter)
            .sum()
            .reset_index()
            .rename(columns={"duration": "total_hours"})
        )
        costs["display_reason"] = [""] * costs.shape[0]

        non_shift_hours = (
            "not in working hours",
            "work shift has ended; waiting for next shift to start",
            "no more return visits will be made",
            "will return next year",
            "waiting for next operational period",
            "end of shift; will resume work in the next shift",
        )
        weather_hours = (
            "weather delay",
            "weather unsuitable to transfer crew",
            "insufficient time to complete travel before end of the shift",
            "weather unsuitable for mooring reconnection",
            "weather unsuitable for unmooring",
        )
        costs.loc[
            (costs.action == "delay") & (costs.additional.isin(non_shift_hours)),
            "display_reason",
        ] = "Not in Shift"
        costs.loc[costs.action == "repair", "display_reason"] = "Repair"
        costs.loc[costs.action == "maintenance", "display_reason"] = "Maintenance"
        costs.loc[costs.action == "transferring crew", "display_reason"] = (
            "Crew Transfer"
        )
        costs.loc[costs.action == "traveling", "display_reason"] = "Site Travel"
        costs.loc[costs.action == "towing", "display_reason"] = "Towing"
        costs.loc[costs.action.str.contains("mooring"), "display_reason"] = "Connection"
        costs.loc[costs.action == "mobilization", "display_reason"] = "Mobilization"
        costs.loc[costs.additional.isin(weather_hours), "display_reason"] = (
            "Weather Delay"
        )
        costs.loc[costs.reason == "no requests", "display_reason"] = "No Requests"

        costs.reason = costs.display_reason

        drop_columns = [self._materials_cost, "display_reason", "additional", "action"]
        if not by_category:
            drop_columns.extend(
                [
                    self._hourly_cost,
                    self._salary_cost,
                    self._labor_cost,
                    self._equipment_cost,
                ]
            )
        group_filter.pop(group_filter.index("additional"))
        group_filter.pop(group_filter.index("action"))
        costs = costs.drop(columns=drop_columns)
        costs = costs.groupby(group_filter).sum().reset_index()

        comparison_values: product[tuple[Any, Any]] | product[tuple[Any, Any, Any]]
        month_year = frequency == "month-year"
        if frequency in ("annual", "month-year"):
            years = costs.year.unique()
            reasons = costs.reason.unique()
            comparison_values = product(years, reasons)
            if month_year:
                months = costs.month.unique()
                comparison_values = product(years, months, reasons)

            zeros = np.zeros(costs.shape[1] - 2).tolist()
            for _year, *_month, _reason in comparison_values:
                row_filter = costs.year.values == _year
                row = [_year, _reason] + zeros
                if month_year:
                    _month = _month[0]
                    row_filter &= costs.month.values == _month
                    row = [_year, _month, _reason] + zeros[:-1]

                row_filter &= costs.reason.values == _reason
                if costs.loc[row_filter].size > 0:
                    continue
                costs.loc[costs.shape[0]] = row
        elif frequency is Frequency.MONTHLY:
            months = costs.month.unique()
            reasons = costs.reason.unique()
            comparison_values = product(months, reasons)
            zeros = np.zeros(costs.shape[1] - 2).tolist()
            for _month, _reason in comparison_values:
                row_filter = costs.month.values == _month
                row_filter &= costs.reason.values == _reason
                row = [_month, _reason] + zeros
                if costs.loc[row_filter].size > 0:
                    continue
                costs.loc[costs.shape[0]] = row

        new_sort = [
            "Maintenance",
            "Repair",
            "Crew Transfer",
            "Site Travel",
            "Towing",
            "Mobilization",
            "Weather Delay",
            "No Requests",
            "Not in Shift",
        ]
        costs.reason = pd.Categorical(costs.reason, new_sort)
        costs = costs.set_index(group_filter)
        sort_order = ["reason"]
        if by_equipment:
            costs = costs.loc[costs.index.get_level_values("agent").isin(equipment)]
            costs.index = costs.index.set_names({"agent": "equipment_name"})
            sort_order = ["equipment_name", "reason"]
        sort_order = frequency.group_cols + sort_order
        return costs.sort_values(by=sort_order)

    def emissions(
        self,
        emissions_factors: dict,
        maneuvering_factor: float = 0.1,
        port_engine_on_factor: float = 0.25,
    ) -> pd.DataFrame:
        """Calculates the emissions, typically in tons, per hour of operations for
        transiting, maneuvering (calculated as a % of transiting), idling at the site
        (repairs, crew transfer, weather delays), and idling at port (weather delays),
        excluding waiting overnight between shifts.

        Parameters
        ----------
        emissions_factors : dict
            Dictionary of emissions per hour for "transit", "maneuver", "idle at site",
            and "idle at port" for each of the servicing equipment in the simulation.
        maneuvering_factor : float, optional
            The proportion of transit time that can be attributed to
            maneuvering/positioning, by default 0.1.
        port_engine_on_factor : float, optional
            The proportion of idling at port time that can be attributed to having the
            engine on and producing emissions, by default 0.25.

        Returns
        -------
        pd.DataFrame
            DataFrame of "duration" (hours), "distance_km", and "emissions" (tons) for
            each servicing equipment in the simulation for each emissions category.

        Raises
        ------
        KeyError
            Raised if any of the servicing equipment are missing from the
            ``emissions_factors`` dictionary.
        KeyError
            Raised if any of the emissions categories are missing from each servcing
            equipment definition in ``emissions_factors``.
        """
        if missing := set(self.service_equipment_names).difference(
            [*emissions_factors]
        ):
            raise KeyError(
                f"`emissions_factors` is missing the following keys: {missing}"
            )

        valid_categories = ("transit", "maneuvering", "idle at port", "idle at site")
        emissions_categories = list(
            chain(*[[*val] for val in emissions_factors.values()])
        )
        emissions_input = Counter(emissions_categories)
        if (
            len(set(valid_categories).difference(emissions_input.keys())) > 0
            or len(set(emissions_input.values())) > 1
        ):
            raise KeyError(
                "Each servicing equipment's emissions factors must have inputs for:"
                f"{valid_categories}"
            )

        # Create the agent/duration subset
        equipment_usage = (
            self.events.loc[
                self.events.agent.isin(self.service_equipment_names),
                ["agent", "action", "reason", "location", "duration", "distance_km"],
            ]
            .groupby(["agent", "action", "reason", "location"])
            .sum()
            .reset_index(drop=False)
        )
        equipment_usage = equipment_usage.loc[
            ~(
                (equipment_usage.action == "delay")
                & equipment_usage.reason.isin(("no requests", "work is complete"))
            )
        ]

        # Map each of the locations to new categories and filter out unnecessary ones
        conditions = [
            equipment_usage.location.eq("site").astype(bool),
            equipment_usage.location.eq("system").astype(bool),
            equipment_usage.location.eq("port").astype(bool),
            equipment_usage.location.eq("enroute").astype(bool),
        ]
        values = ["idle at site", "idle at site", "idle at port", "transit"]
        equipment_usage = (
            equipment_usage.assign(
                category=np.select(conditions, values, default="invalid")
            )
            .drop(["action", "reason", "location"], axis=1)
            .groupby(["agent", "category"])
            .sum()
            .drop("invalid", level="category")
        )

        # Create a new emissions factor DataFrame and mapping
        categories = list(set().union(emissions_categories))
        emissions_summary = pd.DataFrame(
            [],
            index=pd.MultiIndex.from_product(
                [[*emissions_factors], categories], names=["agent", "category"]
            ),
        )
        factors = [
            [(eq, cat), ef]
            for eq, d in emissions_factors.items()
            for cat, ef in d.items()
        ]
        emissions_summary.loc[[ix for (ix, _) in factors], "emissions_factors"] = [
            ef for (_, ef) in factors
        ]

        # Combine the emissions factors and the calculate the total distribution
        equipment_usage = equipment_usage.join(emissions_summary, how="outer").fillna(0)

        # Adjust the transiting time to account for maneuvering
        transiting = equipment_usage.index.get_level_values("category") == "transit"
        manuevering = (
            equipment_usage.index.get_level_values("category") == "maneuvering"
        )
        equipment_usage.loc[manuevering, "duration"] = (
            equipment_usage.loc[transiting, "duration"].values * maneuvering_factor
        )
        equipment_usage.loc[transiting, "duration"] = equipment_usage.loc[
            transiting, "duration"
        ] * (1 - maneuvering_factor)

        # Adjust the idling at port time to only account for when the engine is on
        port = equipment_usage.index.get_level_values("category") == "idle at port"
        equipment_usage.loc[port, "duration"] = (
            equipment_usage.loc[transiting, "duration"].values * port_engine_on_factor
        )

        equipment_usage = (
            equipment_usage.fillna(0)
            .assign(
                emissions=equipment_usage.duration * equipment_usage.emissions_factors
            )
            .drop(columns=["emissions_factors"])
            .fillna(0, axis=1)
        )

        return equipment_usage

    def component_costs(
        self,
        frequency: str,
        by_category: bool = False,
        by_action: bool = False,
        by_task: bool = False,
        *,
        include_travel: bool = False,
    ) -> pd.DataFrame:
        """Calculates the component costs for the simulation at a project, annual, or
        monthly level that can be broken out by cost categories. This will not sum to
        the total cost because it is does not include times where there is no work being
        done, but costs are being accrued.

        .. note:: It should be noted that the costs will include costs accrued from both
           weather delays and shift-to-shift delays. In the future these will be
           disentangled.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by_category : bool, optional
            Indicates whether the values are with resepect to the various cost
            categories (True) or not (False), by default False.
        by_action : bool, optional
            Indicates whether component costs are going to be further broken out by the
            action being performed--repair, maintenance, and delay--(True) or not
            (False), by default False.
        by_task : bool, optional
            Indicates if each repair or maintenance task type should be broken out for
            each of the components, by default False.
        include_travel : bool, optional
            Indicates if travel costs associated with this repair should be included, by
            default False.

        Returns
        -------
        float | pd.DataFrame
            Returns either a float for whole project-level costs or a pandas
            ``DataFrame`` with columns:

            - year (if appropriate for frequency)
            - month (if appropriate for frequency)
            - subassembly
            - task (if broken out)
            - action (if broken out)
            - materials_cost (if broken out)
            - total_labor_cost (if broken out)
            - equipment_cost (if broken out)
            - total_cost

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or
            "month-year".
        ValueError
            If :py:attr:`by_category` is not one of ``True`` or ``False``.
        ValueError
            If :py:attr:`by_action` is not one of ``True`` or ``False``.
        ValueError
            If :py:attr:`by_task` is not one of ``True`` or ``False``.
        ValueError
            If :py:attr:`include_travel` is not one of ``True`` or ``False``.
        """
        frequency = _check_frequency(frequency, which="all")
        if not isinstance(by_category, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")
        if not isinstance(by_action, bool):
            raise ValueError("``by_action`` must be one of ``True`` or ``False``")
        if not isinstance(by_task, bool):
            raise ValueError("``by_task`` must be one of ``True`` or ``False``")
        if not isinstance(include_travel, bool):
            raise ValueError("``include_travel`` must be one of ``True`` or ``False``")

        cost_cols = [self._total_cost]
        if by_category:
            cost_cols[0:0] = [
                self._materials_cost,
                self._labor_cost,
                self._equipment_cost,
            ]

        part_group = (
            self.events.loc[
                (
                    self.events.action.eq("repair request")
                    | self.events.action.eq("maintenance request")
                )
                & self.events.request_id.str.startswith(("RPR", "MNT")),
                ["agent", "reason", "request_id", "system_name"],
            ]
            .drop_duplicates(subset=["request_id"], keep="first")
            .groupby(["agent", "reason", "request_id"])
            .count()
            .reset_index(level=["agent", "reason"], drop=False)
            .drop(columns="system_name")
            .sort_index()
            .rename(columns={"agent": "subassembly", "reason": "task"})
        )

        group_filter = frequency.group_cols
        group_filter.append("subassembly")
        if by_task:
            group_filter.append("task")
        if by_action:
            group_filter.append("action")

        cost_group = (
            self.events.loc[
                self.events.request_id.str.startswith(("RPR", "MNT"))
                & self.events[self._total_cost].gt(0),
                ["year", "month", "request_id", "action", *cost_cols],
            ]
            .replace(
                {
                    "repair complete": "repair",
                    "maintenance complete": "maintenance",
                    "transferring crew": "travel",
                    "traveling": "travel",
                }
            )
            .groupby(["year", "month", "request_id", "action"])
            .sum()
            .reset_index(["year", "month", "action"], drop=False)
            .sort_index()
        )
        if not include_travel:
            cost_group = cost_group.loc[cost_group.action.ne("travel")]

        component_costs = (
            part_group.join(cost_group, how="outer")
            .loc[:, [*group_filter, *cost_cols]]
            .reset_index(drop=True)
            .convert_dtypes()
            .groupby(group_filter)
            .sum()
            .sort_index(level=group_filter)
        )
        return component_costs

    def port_fees(self, frequency: str) -> pd.DataFrame:
        """Calculates the port fees for the simulation at a project, annual, or monthly
        level. This excludes any equipment or labor costs, which are included in:
        ``equipment_costs``.

        Parameters
        ----------
        frequency : str
            One of "project" or "annual", "monthly", ".

        Returns
        -------
        pd.DataFrame
            The broken out by time port fees with

        Raises
        ------
        ValueError
            If ``frequency`` not one of "project" or "annual".
        """
        frequency = _check_frequency(frequency, which="all")

        column = "port_fees"
        port_fee = self.events.loc[
            self.events.action == "monthly lease fee",
            ["year", "month", "equipment_cost"],
        ].rename(columns={"equipment_cost": column})

        if port_fee.shape[0] == 0:
            return pd.DataFrame([[0]], columns=[column])

        if frequency is Frequency.PROJECT:
            return pd.DataFrame([port_fee.sum(axis=0).loc[column]], columns=[column])

        group_filter = frequency.group_cols
        return port_fee[group_filter + [column]].groupby(group_filter).sum()

    def project_fixed_costs(self, frequency: str, resolution: str) -> pd.DataFrame:
        """Calculates the fixed costs of a project at the project and annual frequencies
        at a given cost breakdown resolution.

        Parameters
        ----------
        frequency : str
            One of "project" or "annual", "monthly", ".
        resolution : st
            One of "low", "medium", or "high", where the values correspond to:

            - low: ``FixedCosts.resolution["low"]``, corresponding to itemized costs.
            - medium: ``FixedCosts.resolution["medium"]``, corresponding to the
              overarching cost categories.
            - high: ``FixedCosts.resolution["high"]``, corresponding to a lump sum.

            These values can also be seen through the ``FixedCosts.hierarchy``

        Returns
        -------
        pd.DataFrame
            The project's fixed costs as a sum or annualized with high, medium, and low
            resolution as desired.

        Raises
        ------
        ValueError
            If ``frequency`` not one of "project" or "annual".
        ValueError
            If ``resolution`` must be one of "low", "medium", or "high".
        """
        frequency = _check_frequency(frequency, which="all")

        resolution = resolution.lower().strip()
        if resolution not in ("low", "medium", "high"):
            raise ValueError(
                '``resolution`` must be one of "low", "medium", or "high".'
            )

        # Get the appropriate values and convert to the currency base
        keys = self.fixed_costs.resolution[resolution]
        vals = (
            np.array([[getattr(self.fixed_costs, key) for key in keys]])
            * self.project_capacity
            * 1000
        )

        total = (
            self.operations[["year", "month", "env_time"]]
            .groupby(["year", "month"])
            .count()
        )
        total = total.rename(columns={"env_time": "N"})
        total.N = 1.0

        operation_hours = (
            self.operations[["year", "month", "env_time"]]
            .groupby(["year", "month"])
            .count()
        )
        operation_hours = operation_hours.rename(columns={"env_time": "N"})

        costs = pd.DataFrame(total.values * vals, index=total.index, columns=keys)
        costs *= operation_hours.values.reshape(-1, 1) / 8760.0

        adjusted_inflation = np.array(
            [self.inflation_rate ** (i // 12) for i in range(costs.shape[0])]
        )
        costs *= adjusted_inflation.reshape(-1, 1)

        if frequency is Frequency.PROJECT:
            return pd.DataFrame(costs.reset_index(drop=True).sum()).T

        costs = (
            costs.reset_index(drop=False)
            .drop(columns=frequency.drop_cols)
            .groupby(frequency.group_cols)
            .sum()
        )

        return costs

    def opex(self, frequency: str, by_category: bool = False) -> pd.DataFrame:
        """Calculates the project's OpEx for the simulation at a project, annual, or
        monthly level.

        Parameters
        ----------
        frequency : str
            One of project, annual, monthly, or month-year.

        by_category : bool, optional
            Indicates whether the values are with resepect to the various cost
            categories (True) or not (False), by default False.

        Returns
        -------
        pd.DataFrame
            The project's OpEx broken out at the desired time and category resolution.
        """
        frequency = _check_frequency(frequency, which="all")

        # Get the materials costs and remove the component-level breakdown
        materials = self.component_costs(frequency=frequency, by_category=True)
        materials = materials.loc[:, ["materials_cost"]].reset_index()
        if frequency is Frequency.PROJECT:
            materials = pd.DataFrame(materials.loc[:, ["materials_cost"]].sum()).T
        else:
            group_col = frequency.group_cols
            materials = (
                materials[group_col + ["materials_cost"]].groupby(group_col).sum()
            )

        # Port fees will produce an 1x1 dataframe if values aren't present, so recreate
        # it with the appropriate dimension
        port_fees = self.port_fees(frequency=frequency)
        if frequency != "project" and port_fees.shape == (1, 1):
            port_fees = pd.DataFrame(
                [], columns=["port_fees"], index=materials.index, dtype=float
            )
            port_fees = port_fees.astype(float).fillna(0)

        # Create a list of data frames for the OpEx components
        opex_items = [
            self.project_fixed_costs(frequency=frequency, resolution="low"),
            port_fees,
            self.equipment_costs(frequency=frequency),
            self.labor_costs(frequency=frequency),
            materials,
        ]

        # Join the data frames and sum along the time axis and return
        column = "OpEx"
        opex = pd.concat(opex_items, axis=1)
        opex.loc[:, column] = opex.sum(axis=1)
        if by_category:
            return opex
        return opex[[column]]

    def process_times(self, include_incompletes: bool = True) -> pd.DataFrame:
        """Calculates the time, in hours, to complete a repair/maintenance request, on
        both a request to completion basis, and the actual time to complete the repair.

        Parameters
        ----------
        include_incompletes : bool, optional
            Boolean flag to include the incomplete repair and maintenance requests. If
            True, this will summarize all submitted requests that have not been
            canceled, but if False, this will only summary the process timing for
            the completed requests, by default True.

        Returns
        -------
        pd.DataFrame
         - category (index): repair/maintenance category
         - time_to_completion: total number of hours from the time of request to the
           time of completion
         - process_time: total number of hours it took for the equipment to complete
         - the request.
         - downtime: total number of hours where the operations were below 100%.
         - N: total number of processes in the category.
        """
        canceled_requests = self.events.loc[
            self.events.action.isin(("repair canceled", "maintenance canceled")),
            "request_id",
        ]
        events_valid = self.events.loc[
            self.events.request_id.ne("na")
            & ~self.events.request_id.isin(canceled_requests)
        ]
        if not include_incompletes:
            completed = self.events.loc[
                self.events.action.isin(("repair complete", "maintenance complete")),
                "request_id",
            ]
            events_valid = events_valid.loc[events_valid.request_id.isin(completed)]

        # Summarize all the requests data
        request_df = (
            events_valid[["request_id", "env_time", "duration"]]
            .groupby("request_id")
            .sum()
            .sort_index()
        )
        request_df_min = (
            events_valid[["request_id", "env_time", "duration"]]
            .groupby("request_id")
            .min()
            .sort_index()
        )
        request_df_max = (
            events_valid[["request_id", "env_time", "duration"]]
            .groupby("request_id")
            .max()
            .sort_index()
        )

        # Summarize all the downtime-specific data for all requests
        downtime_df = events_valid.loc[events_valid.system_operating_level < 1][
            ["request_id", "env_time", "duration"]
        ]
        downtime_df_min = (
            downtime_df[["request_id", "env_time", "duration"]]
            .groupby("request_id")
            .min()
            .sort_index()
        )
        downtime_df_max = (
            downtime_df[["request_id", "env_time", "duration"]]
            .groupby("request_id")
            .max()
            .sort_index()
        )

        reason_df = (
            events_valid.drop_duplicates(subset=["request_id"])[
                ["request_id", "part_name", "reason"]
            ]
            .set_index("request_id")
            .sort_index()
        )

        # Summarize the time to first repair/maintenance activity
        submitted_df = (
            events_valid.loc[
                events_valid.action.isin(("repair request", "maintenance request")),
                ["request_id", "env_time"],
            ]
            .drop_duplicates(subset=["request_id"])
            .set_index("request_id")
            .sort_index()
        )
        action_df = (
            events_valid.loc[
                events_valid.action.isin(("repair", "maintenance")),
                ["request_id", "env_time"],
            ]
            .groupby("request_id")
            .min()
            .sort_index()
        )
        time_to_repair_df = action_df.subtract(submitted_df, axis="index")

        # Create the timing dataframe
        timing = pd.DataFrame([], index=request_df_min.index)
        timing = timing.join(reason_df[["part_name", "reason"]]).rename(
            columns={"part_name": "subassembly", "reason": "task"}
        )
        timing = timing.join(
            request_df_min[["env_time"]]
            .join(request_df_max[["env_time"]], lsuffix="_min", rsuffix="_max")
            .diff(axis=1)[["env_time_max"]]
            .rename(columns={"env_time_max": "time_to_completion"})
        )
        timing = timing.join(request_df[["duration"]]).rename(
            columns={"duration": "process_time"}
        )
        timing = timing.join(
            downtime_df_min[["env_time"]]
            .join(downtime_df_max[["env_time"]], lsuffix="_min", rsuffix="_max")
            .diff(axis=1)[["env_time_max"]]
            .rename(columns={"env_time_max": "downtime"})
        )
        timing = timing.join(
            time_to_repair_df.rename(columns={"env_time": "time_to_start"})
        )
        timing["N"] = 1

        # Return only the categorically summed data
        return timing.groupby(["subassembly", "task"]).sum().sort_index()

    def request_summary(self) -> pd.DataFrame:
        """Calculate the number of repair and maintenance requets that have been
        submitted, cancelled, not completed, and completed.

        Returns
        -------
        pd.DataFrame
            Data frame with a :py:class`pandas.MultiIndex` of the subassembly and
            task description, with columns "total_request", "canceled_request",
            "incomplete_requests", and "completed_requests".
        """
        requests = self.events.loc[
            self.events.action.isin(("repair request", "maintenance request")),
            "request_id",
        ].drop_duplicates()
        canceled_requests = self.events.loc[
            self.events.action.isin(("repair canceled", "maintenance canceled")),
            "request_id",
        ].drop_duplicates()
        completed_requests = self.events.loc[
            self.events.action.isin(("repair complete", "maintenance complete")),
            "request_id",
        ].drop_duplicates()
        incomplete_requests = requests.loc[
            ~requests.isin(canceled_requests) & ~requests.isin(completed_requests)
        ]
        total_df = (
            self.events.loc[
                self.events.action.isin(("repair request", "maintenance request")),
                ["part_name", "reason", "request_id"],
            ]
            .rename(
                columns={
                    "part_name": "subassembly",
                    "reason": "task",
                }
            )
            .drop_duplicates(subset=["request_id"])
        )
        canceled_df = (
            total_df.loc[total_df.request_id.isin(canceled_requests)]
            .groupby(["subassembly", "task"])
            .count()
            .rename(columns={"request_id": "canceled_requests"})
        )
        incomplete_df = (
            total_df.loc[total_df.request_id.isin(incomplete_requests)]
            .groupby(["subassembly", "task"])
            .count()
            .rename(columns={"request_id": "incomplete_requests"})
        )
        completed_df = (
            total_df.loc[total_df.request_id.isin(completed_requests)]
            .groupby(["subassembly", "task"])
            .count()
            .rename(columns={"request_id": "completed_requests"})
        )
        total_df = (
            total_df.groupby(["subassembly", "task"])
            .count()
            .rename(columns={"request_id": "total_requests"})
        )
        summary = (
            total_df.join(canceled_df, how="outer")
            .join(incomplete_df, how="outer")
            .join(completed_df, how="outer")
            .fillna(0)
            .astype(int)
        )
        return summary

    def power_production(
        self, frequency: str, by: str = "windfarm", units: str = "gwh"
    ) -> float | pd.DataFrame:
        """Calculates the power production for the simulation at a project, annual, or
        monthly level that can be broken out by turbine.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by : str
            One of "windfarm" or "turbine".
        units : str
            One of "gwh", "mwh", or "kwh".

        Returns
        -------
        float | pd.DataFrame
            Returns either a float for whole project-level energy production or a pandas
            ``DataFrame`` with columns:

            - year (if appropriate for frequency)
            - month (if appropriate for frequency)
            - total_power_production
            - <turbine_id>_power_production (if broken out)

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or
            "month-year".
        ValueError
            If :py:attr:`by` is not one of "turbine" or "windfarm".
        """
        frequency = _check_frequency(frequency, which="all")

        by = by.lower().strip()
        if by not in ("windfarm", "turbine"):
            raise ValueError('``by`` must be one of "windfarm" or "turbine".')
        by_turbine = by == "turbine"

        if units not in ("gwh", "mwh", "kwh"):
            raise ValueError('``units`` must be one of "gwh", "mwh", or "kwh".')
        if units == "gwh":
            divisor = 1e6
            label = "Project Energy Production (GWh)"
        elif units == "mwh":
            divisor = 1e3
            label = "Project Energy Production (MWh)"
        else:
            divisor = 1
            label = "Project Energy Production (kWh)"

        col_filter = ["windfarm"]
        if by_turbine:
            col_filter.extend(self.turbine_id)

        if frequency is Frequency.PROJECT:
            production = self.production[col_filter].sum(axis=0)
            production = (
                pd.DataFrame(
                    production.values.reshape(1, -1),
                    columns=col_filter,
                    index=[label],
                )
                / divisor
            )
            return production

        group_cols = frequency.group_cols
        production = (
            self.production[group_cols + col_filter].groupby(by=group_cols).sum()
            / divisor
        )
        return production

    def h2_production(
        self, frequency: str, by: str = "total", units: str = "kg"
    ) -> float | pd.DataFrame:
        """Calculates the hydrogen production for the simulation at a project, annual,
        or monthly level that can be broken out by electrolyzer.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by : str
            One of "electrolyzer" or "total".
        units : str
            One of "kg" (kilograms) or "tn" (metric tonnes).

        Returns
        -------
        float | pd.DataFrame
            Returns either a float for whole project-level hydrogen production or a
            pandas ``DataFrame`` with columns:

            - year (if appropriate for frequency)
            - month (if appropriate for frequency)
            - total_power_production
            - <electrolyzer>_power_production (if broken out)

        Raises
        ------
        ValueError
            Raised if there were no simulated electrolyzers.
        ValueError
            If :py:attr:`frequency` is not one of "project", "annual", "monthly", or
            "month-year".
        ValueError
            If :py:attr:`by` is not one of "electrolyzer" or "total".
        ValueError
            If :py:attr:`units` is not one of "kg" or "tn".
        """
        if self.electrolyzer_rated_production.size == 0:
            raise ValueError("No electrolyzers available to analyze.")
        frequency = _check_frequency(frequency, which="all")

        by = by.lower().strip()
        if by not in ("electrolyzer", "total"):
            raise ValueError('``by`` must be one of "total" or "electrolyzer".')
        by_electrolyzer = by == "electrolyzer"

        if units not in ("kg", "tn"):
            raise ValueError('``units`` must be one of "kg" or "tn".')
        if units == "tn":
            divisor = 1e3
            label = "Project H2 Production (metric tonnes)"
        else:
            divisor = 1
            label = "Project H2 Production (kg)"

        col_filter = ["total"]
        if by_electrolyzer:
            col_filter.extend(self.electrolyzer_id)

        production = self.production.copy()
        production["total"] = production[self.electrolyzer_id].sum(axis=1)

        if frequency is Frequency.PROJECT:
            production = production[col_filter].sum(axis=0)
            production = (
                pd.DataFrame(
                    production.values.reshape(1, -1),
                    columns=col_filter,
                    index=[label],
                )
                / divisor
            )
            return production

        group_cols = frequency.group_cols
        production = (
            production[group_cols + col_filter].groupby(by=group_cols).sum() / divisor
        )
        return production

    # Windfarm Financials

    def npv(
        self, frequency: str, discount_rate: float = 0.025, offtake_price: float = 80
    ) -> pd.DataFrame:
        """Calculates the net present value of the windfarm at a project, annual, or
        monthly resolution given a base discount rate and offtake price.

        .. note:: This function will be improved over time to incorporate more of the
            financial parameter at play, such as PPAs.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        discount_rate : float, optional
            The rate of return that could be earned on alternative investments, by
            default 0.025.
        offtake_price : float, optional
            Price of energy, per MWh, by default 80.

        Returns
        -------
        pd.DataFrame
            The project net prsent value at the desired time resolution.
        """
        frequency = _check_frequency(frequency, which="all")

        # Gather the OpEx, and revenues
        expenditures = self.opex("month-year")
        production = self.power_production("month-year")
        revenue: pd.DataFrame = production / 1000 * offtake_price  # MWh

        # Instantiate the NPV with the required calculated data and compute the result
        npv = revenue.join(expenditures).rename(columns={"windfarm": "revenue"})
        N = npv.shape[0]
        npv.loc[:, "discount"] = np.full(N, 1 + discount_rate) ** np.arange(N)
        npv.loc[:, "NPV"] = (npv.revenue.values - npv.OpEx.values) / npv.discount.values

        # Aggregate the results to the required resolution
        if frequency is Frequency.PROJECT:
            return pd.DataFrame(npv.reset_index().sum()).T[["NPV"]]

        if frequency is Frequency.MONTH_YEAR:
            return npv[["NPV"]]

        npv = (
            npv.reset_index()
            .drop(columns=frequency.drop_cols)
            .groupby(frequency.group_cols)
            .sum()[["NPV"]]
        )
        return npv
