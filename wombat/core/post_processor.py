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

from wombat.core import FixedCosts
from wombat.utilities import calculate_windfarm_operational_level
from wombat.core.library import load_yaml


def _check_frequency(frequency: str, which: str = "all") -> str:
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
    str
        The lower-case, input with white spaces removed.

    Raises
    ------
    ValueError
        Raised if an invalid value was raised
    """
    opts: tuple[str, ...]
    if which == "all":
        opts = ("project", "annual", "monthly", "month-year")
    elif which == "monthly":
        opts = ("project", "annual", "monthly")
    elif which == "annual":
        opts = ("project", "annual")
    frequency = frequency.lower().strip()
    if frequency not in opts:
        raise ValueError(f"``frequency`` must be one of {opts}.")
    return frequency


def _calculate_time_availability(
    availability: pd.DataFrame,
    by_turbine: bool = False,
) -> float | np.ndarray:
    """Calculates the availability ratio of the whole timeseries or the whole
    timeseries, by turbine.

    Parameters
    ----------
    availability : pd.DataFrame
        Timeseries array of operating ratios for all turbines.
    by_turbine : bool, optional
        If True, calculates the availability rate of each column, otherwise across the
        whole array, by default False.

    Returns
    -------
    float | np.ndarray
        Availability ratio across the whole timeseries, or broken out by column
        (turbine).
    """
    availability = availability > 0
    if by_turbine:
        return availability.values.sum(axis=0) / availability.shape[0]
    return availability.values.sum() / availability.size


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
        substation_id: str | list[str],
        turbine_id: str | list[str],
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
            Either a pandas ``DataFrame`` or filename to be used to read the csv log
            data.
        operations : str | pd.DataFrame
            Either a pandas ``DataFrame`` or filename to be used to read the csv log
            data.
        potential : str | pd.DataFrame
            Either a pandas ``DataFrame`` or a filename to be used to read the csv
            potential power production data.
        production : str | pd.DataFrame
            Either a pandas ``DataFrame`` or a filename to be used to read the csv power
            production data.
        inflation_rate : float
            The inflation rate to be applied to all dollar amounts from the analysis
            starting year to ending year.
        project_capacity : float
            The project's rated capacity, in MW.
        turbine_capacities : Union[float, List[float]]
            The capacity of each individual turbine corresponding to ``turbine_id``, in
            kW.
        substation_id : str | list[str]
            The substation id(s).
        turbine_id : str | list[str]
            The turbine id(s).
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
        self.turbine_capacities = turbine_capacities

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

    def __eq__(self, other) -> bool:
        """Check that the essential information is the same."""
        if isinstance(other, Metrics):
            return all(
                expected.equals(actual)
                if isinstance(expected, pd.DataFrame)
                else expected == actual
                for _, expected, actual in self._yield_comparisons(other)
            )
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
            "substation_id",
            "fixed_costs",
            "turbine_capacities",
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
        """Tidies the "raw" csv-converted data to be able to be used among the
        ``Metrics`` class.

        Parameters
        ----------
        data : pd.DataFrame
            The csv log data.

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
        """Reads the csv log data from library. This is intended to be used for the
        events or operations data.

        Parameters
        ----------
        path : str
            Path to the simulation library.
        fname : str
            Filename of the csv data.

        Returns
        -------
        pd.DataFrame
            Dataframe of either the events or operations data.
        """
        if "events" in fname:
            data = (
                pd.read_csv(
                    self.data_dir / "outputs" / "logs" / fname,
                    delimiter="|",
                    engine="pyarrow",
                    dtype={
                        "agent": "string",
                        "action": "string",
                        "reason": "string",
                        "additional": "string",
                        "system_id": "string",
                        "system_name": "string",
                        "part_id": "string",
                        "part_name": "string",
                        "request_id": "string",
                        "location": "string",
                    },
                )
                .set_index("datetime")
                .sort_index()
            )
            return data

        data = pd.read_csv(
            self.data_dir / "outputs" / "logs" / fname,
            delimiter="|",
            engine="pyarrow",
        )
        return data

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
        """Calculates the time-based availabiliy over a project's lifetime as a single
        value, annual average, or monthly average for the whole windfarm or by turbine.

        .. note:: This currently assumes that if there are multiple substations, that
          the turbines are all connected to multiple.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by : str
            One of "windfarm" or "turbine".

        Returns
        -------
        pd.DataFrame
            The time-based availability at the desired aggregation level.
        """
        frequency = _check_frequency(frequency, which="all")

        by = by.lower().strip()
        if by not in ("windfarm", "turbine"):
            raise ValueError('``by`` must be one of "windfarm" or "turbine".')
        by_turbine = by == "turbine"

        # Determine the operational capacity of each turbine with substation downtime
        operations_cols = ["year", "month", "day", "windfarm"] + self.turbine_id
        turbine_operations = self.operations[operations_cols].copy()

        hourly = turbine_operations.loc[:, self.turbine_id]

        # TODO: The below should be better summarized as:
        # (availability > 0).groupby().sum() / groupby().count()

        if frequency == "project":
            availability = _calculate_time_availability(hourly, by_turbine=by_turbine)
            if not by_turbine:
                return pd.DataFrame([availability], columns=["windfarm"])

            if TYPE_CHECKING:
                assert isinstance(availability, np.ndarray)
            availability = pd.DataFrame(
                availability.reshape(1, -1), columns=self.turbine_id
            )
            return availability
        elif frequency == "annual":
            date_time = turbine_operations[["year"]]
            counts = turbine_operations.groupby(by="year").count()
            counts = counts[self.turbine_id] if by_turbine else counts[["windfarm"]]
            annual = [
                _calculate_time_availability(
                    hourly[date_time.year == year],
                    by_turbine=by_turbine,
                )
                for year in counts.index
            ]
            return pd.DataFrame(annual, index=counts.index, columns=counts.columns)
        elif frequency == "monthly":
            date_time = turbine_operations[["month"]]
            counts = turbine_operations.groupby(by="month").count()
            counts = counts[self.turbine_id] if by_turbine else counts[["windfarm"]]
            monthly = [
                _calculate_time_availability(
                    hourly[date_time.month == month],
                    by_turbine=by_turbine,
                )
                for month in counts.index
            ]
            return pd.DataFrame(monthly, index=counts.index, columns=counts.columns)
        elif frequency == "month-year":
            date_time = turbine_operations[["year", "month"]]
            counts = turbine_operations.groupby(by=["year", "month"]).count()
            counts = counts[self.turbine_id] if by_turbine else counts[["windfarm"]]
            month_year = [
                _calculate_time_availability(
                    hourly[(date_time.year == year) & (date_time.month == month)],
                    by_turbine=by_turbine,
                )
                for year, month in counts.index
            ]
            return pd.DataFrame(month_year, index=counts.index, columns=counts.columns)

    def production_based_availability(self, frequency: str, by: str) -> pd.DataFrame:
        """Calculates the production-based availabiliy over a project's lifetime as a
        single value, annual average, or monthly average for the whole windfarm or by
        turbine.

        .. note:: This currently assumes that if there are multiple substations, that
          the turbines are all connected to multiple.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by : str
            One of "windfarm" or "turbine".

        Returns
        -------
        pd.DataFrame
            The production-based availability at the desired aggregation level.
        """
        frequency = _check_frequency(frequency, which="all")

        by = by.lower().strip()
        if by not in ("windfarm", "turbine"):
            raise ValueError('``by`` must be one of "windfarm" or "turbine".')
        by_turbine = by == "turbine"

        if by_turbine:
            production = self.production.loc[:, self.turbine_id]
            potential = self.potential.loc[:, self.turbine_id]
        else:
            production = self.production[["windfarm"]].copy()
            potential = self.potential[["windfarm"]].copy()

        if frequency == "project":
            production = production.values
            potential = potential.values
            if (potential == 0).sum() > 0:
                potential[potential == 0] = 1

            availability = production.sum(axis=0) / potential.sum(axis=0)
            if by_turbine:
                return pd.DataFrame([availability], columns=self.turbine_id)
            else:
                return pd.DataFrame([availability], columns=["windfarm"])

        production["year"] = production.index.year.values
        production["month"] = production.index.month.values

        potential["year"] = potential.index.year.values
        potential["month"] = potential.index.month.values

        group_cols = deepcopy(self.turbine_id) if by_turbine else ["windfarm"]
        if frequency == "annual":
            group_cols.insert(0, "year")
            production = production[group_cols].groupby("year").sum()
            potential = potential[group_cols].groupby("year").sum()

        elif frequency == "monthly":
            group_cols.insert(0, "month")
            production = production[group_cols].groupby("month").sum()
            potential = potential[group_cols].groupby("month").sum()

        elif frequency == "month-year":
            group_cols.insert(0, "year")
            group_cols.insert(0, "month")
            production = production[group_cols].groupby(["year", "month"]).sum()
            potential = potential[group_cols].groupby(["year", "month"]).sum()

        if (potential.values == 0).sum() > 0:
            potential.loc[potential.values == 0] = 1
        columns = self.turbine_id
        if not by_turbine:
            production = production.sum(axis=1)
            potential = potential.sum(axis=1)
            columns = [by]
        return pd.DataFrame(production / potential, columns=columns)

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
            One of "windfarm" or "turbine".

        Returns
        -------
        pd.DataFrame
            The capacity factor at the desired aggregation level.
        """
        which = which.lower().strip()
        if which not in ("net", "gross"):
            raise ValueError('``which`` must be one of "net" or "gross".')

        frequency = _check_frequency(frequency, which="all")

        by = by.lower().strip()
        if by not in ("windfarm", "turbine"):
            raise ValueError('``by`` must be one of "windfarm" or "turbine".')
        by_turbine = by == "turbine"

        production = self.production if which == "net" else self.potential
        production = production.loc[:, self.turbine_id]

        if frequency == "project":
            if not by_turbine:
                potential = production.shape[0] * self.project_capacity * 1000.0
                production = production.values.sum()
                return pd.DataFrame([production / potential], columns=["windfarm"])

            potential = production.shape[0] * np.array(self.turbine_capacities)
            return pd.DataFrame(production.sum(axis=0) / potential).T

        production["year"] = production.index.year.values
        production["month"] = production.index.month.values

        if frequency == "annual":
            group_cols = ["year"]
        elif frequency == "monthly":
            group_cols = ["month"]
        elif frequency == "month-year":
            group_cols = ["year", "month"]

        potential = production[group_cols + self.turbine_id].groupby(group_cols).count()
        production = production[group_cols + self.turbine_id].groupby(group_cols).sum()

        if by_turbine:
            capacity = np.array(self.turbine_capacities, dtype=float)
            columns = self.turbine_id
            potential *= capacity
        else:
            capacity = self.project_capacity
            production = production.sum(axis=1)
            columns = [by]
        return pd.DataFrame(production / potential, columns=columns)

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

        if frequency == "project":
            if requests.shape[0] == 0:
                return pd.DataFrame([0.0], columns=["windfarm"])
            return pd.DataFrame(
                [completions.shape[0] / requests.shape[0]], columns=["windfarm"]
            )

        requests["year"] = requests.env_datetime.dt.year.values
        requests["month"] = requests.env_datetime.dt.month.values

        completions["year"] = completions.env_datetime.dt.year.values
        completions["month"] = completions.env_datetime.dt.month.values

        if frequency == "annual":
            group_filter = ["year"]
            indices = self.operations.year.unique()
        elif frequency == "monthly":
            group_filter = ["month"]
            indices = self.operations.month.unique()
        elif frequency == "month-year":
            group_filter = ["year", "month"]
            indices = (
                self.operations[["year", "month"]]
                .groupby(["year", "month"])
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

        if frequency == "annual":
            col_filter = ["year"]
        elif frequency == "monthly":
            col_filter = ["month"]
        elif frequency == "month-year":
            col_filter = ["year", "month"]

        cost_col = [self._equipment_cost]
        events = self.events.loc[self.events.action != "monthly lease fee"]
        if by_equipment:
            if frequency == "project":
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
            return costs.fillna(value=0)

        if frequency == "project":
            return pd.DataFrame([events[cost_col].sum()], columns=cost_col)

        costs = events[cost_col + col_filter].groupby(col_filter).sum()
        return costs.fillna(0)

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

        if frequency == "project":
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

        if frequency == "project":
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

        elif frequency == "annual":
            additional_cols = ["year"]
            total_hours = total_hours.groupby("year")[["N"]].sum()
        elif frequency == "monthly":
            additional_cols = ["month"]
            total_hours = total_hours.groupby("month")[["N"]].sum()
        elif frequency == "month-year":
            additional_cols = ["year", "month"]
            total_hours = total_hours.groupby(["year", "month"])[["N"]].sum()

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
        n_tows = towing.groupby(["agent", "year", "month", "direction"]).count()
        n_tows = n_tows.rename(columns={"env_time": "N"})["N"].reset_index()

        # Create a shell for the total tows
        total_tows = (
            self.events[["env_time", "year", "month"]]
            .groupby(["year", "month"])
            .count()["env_time"]
        )
        total_tows = total_tows.reset_index().rename(columns={"env_time": "N"})
        total_tows.N = 0

        # Create the correct time frequency for the number of tows and shell total
        group_cols = ["agent", "direction"]
        if frequency == "project":
            time_cols = []
            n_tows = n_tows[group_cols + ["N"]].groupby(group_cols).sum()

            # If no further work is required, then return the sum as a 1x1 data frame
            if not by_tug and not by_direction:
                return pd.DataFrame(
                    [n_tows.reset_index().N.sum()], columns=["total_tows"]
                )

            total_tows = pd.DataFrame([[0]], columns=["N"])
        elif frequency == "annual":
            time_cols = ["year"]
            columns = time_cols + columns
            group_cols.extend(time_cols)
            n_tows = n_tows.groupby(group_cols).sum()[["N"]]
            total_tows = (
                total_tows[["year", "N"]].groupby(time_cols).sum().reset_index()
            )
        elif frequency == "monthly":
            time_cols = ["month"]
            columns = time_cols + columns
            group_cols.extend(time_cols)
            n_tows = n_tows[group_cols + ["N"]].groupby(group_cols).sum()
            total_tows = (
                total_tows[["month", "N"]].groupby(time_cols).sum().reset_index()
            )
        elif frequency == "month-year":
            # Already have month-year by default, so skip the n_tows refinement
            time_cols = ["year", "month"]
            columns = time_cols + columns
            group_cols.extend(time_cols)
            n_tows = n_tows.set_index(group_cols, drop=True)

        # Create a list of the columns needed for creating the broken down totals
        if frequency == "project":
            total_cols = ["N"]
        else:
            total_cols = total_tows.drop(columns=["N"]).columns.tolist()

        # Sum the number of tows by tugboat, if needed
        if by_tug:
            tug_sums = []
            for tug in tugboats:
                tug_sum = n_tows.loc[tug]
                tug_sums.append(tug_sum.rename(columns={"N": tug}))
            tug_sums_by_direction = pd.concat(tug_sums, axis=1).fillna(0)

            if frequency == "project":
                tug_sums = pd.DataFrame(tug_sums_by_direction.sum()).T
            else:
                tug_sums = tug_sums_by_direction.reset_index().groupby(total_cols).sum()
            if TYPE_CHECKING:
                assert isinstance(tug_sums, pd.DataFrame)  # mypy checking
            tug_sums = tug_sums.rename(
                columns={t: f"{t}_total_tows" for t in tug_sums.columns}
            )
            if TYPE_CHECKING:
                assert isinstance(tug_sums, pd.DataFrame)  # mypy checking
            total = pd.DataFrame(
                tug_sums.sum(axis=1), columns=["total_tows"]
            ).reset_index()
        else:
            if not by_direction:
                # Sum the totals, then merge the results with the shell data frame,
                # and cleanup the columns
                total = n_tows.reset_index().groupby(total_cols).sum().reset_index()
                total_tows = total_tows.merge(total, on=total_cols, how="outer")
                total_tows = total_tows.fillna(0).rename(columns={"N_y": "total_tows"})
                total_tows = total_tows[columns]
                if time_cols:
                    return total_tows.set_index(time_cols)
                return total_tows
            else:
                total = (
                    n_tows.groupby(total_cols)
                    .sum()
                    .reset_index()
                    .rename(columns={"N": "total_tows"})
                )

        # Create the full total tows data
        if frequency == "project":
            if "index" in total.columns:
                total_tows = total.drop(columns=["index"])
        else:
            total_tows = (
                total_tows.merge(total, how="outer").drop(columns=["N"]).fillna(0)
            )
            total_tows = total_tows.set_index(total_cols)

        # Get the sums by each direction towed, if needed
        if by_direction:
            if frequency == "project":
                direction_sums = n_tows.reset_index().groupby("direction").sum()
                for s in direction_suffix:
                    total_tows.loc[:, f"total_tows_{s}"] = direction_sums.loc[s, "N"]
            else:
                direction_sums = (
                    n_tows.reset_index().groupby(["direction"] + total_cols).sum()
                )
                for s in direction_suffix:
                    total_tows = total_tows.join(
                        direction_sums.loc[s].rename(columns={"N": f"total_tows_{s}"})
                    ).fillna(0)

            # Add in the tugboat breakdown as needed
            if by_tug:
                total_tows = total_tows.join(tug_sums, how="outer").fillna(0)
                for s in direction_suffix:
                    if frequency == "project":
                        _total = pd.DataFrame(
                            tug_sums_by_direction.loc[s]
                        ).T.reset_index(drop=True)
                    else:
                        _total = tug_sums_by_direction.loc[s]
                    total_tows = total_tows.join(
                        _total.rename(columns={t: f"{t}_{s}" for t in tugboats}),
                        how="outer",
                    ).fillna(0)
                total_tows = total_tows.reset_index()[columns]
                if time_cols:
                    return total_tows.set_index(time_cols)
                else:
                    return total_tows
            total_tows = total_tows.assign(N=total_tows.sum(axis=1))
            total_tows = total_tows.rename(columns={"N": "total_tows"}).reset_index()[
                columns
            ]
            if time_cols:
                return total_tows.set_index(time_cols)
            return total_tows

        if by_tug:
            total_tows = (
                total_tows.join(tug_sums, how="outer").fillna(0).reset_index()[columns]
            )
            if time_cols:
                return total_tows.set_index(time_cols)
            return total_tows

        if time_cols:
            return total_tows[columns].set_index(time_cols)
        return total_tows[columns]

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
        if frequency == "project":
            costs = pd.DataFrame(
                self.events[labor_cols].sum(axis=0).values.reshape(1, -1),
                columns=labor_cols,
            )
            if not by_type:
                return costs[[self._labor_cost]]
            return costs

        if frequency == "annual":
            group_filter = ["year"]
        elif frequency == "monthly":
            group_filter = ["month"]
        elif frequency == "month-year":
            group_filter = ["year", "month"]

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

        group_filter = ["action", "reason", "additional"]
        if by_equipment:
            group_filter.insert(0, "agent")
        if frequency in ("annual", "month-year"):
            group_filter.insert(0, "year")
        elif frequency == "monthly":
            group_filter.insert(0, "month")
        if frequency == "month-year":
            group_filter.insert(1, "month")

        action_list = [
            "delay",
            "repair",
            "maintenance",
            "mobilization",
            "transferring crew",
            "traveling",
            "towing",
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
        elif frequency == "monthly":
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
        if frequency == "project":
            return costs.sort_values(by=sort_order)
        if frequency == "annual":
            sort_order = ["year"] + sort_order
            return costs.sort_values(by=sort_order)
        if frequency == "monthly":
            sort_order = ["month"] + sort_order
            return costs.sort_values(by=sort_order)
        sort_order = ["year", "month"] + sort_order
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
        self, frequency: str, by_category: bool = False, by_action: bool = False
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

        Returns
        -------
        float | pd.DataFrame
            Returns either a float for whole project-level costs or a pandas
            ``DataFrame`` with columns:

            - year (if appropriate for frequency)
            - month (if appropriate for frequency)
            - component
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
            If ``by_category`` is not one of ``True`` or ``False``.
        ValueError
            If ``by_action`` is not one of ``True`` or ``False``.
        """
        frequency = _check_frequency(frequency, which="all")
        if not isinstance(by_category, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")
        if not isinstance(by_action, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")

        part_filter = ~self.events.part_id.isna() & ~self.events.part_id.isin([""])
        events = self.events.loc[part_filter].copy()

        # Need to simplify the cable identifiers to exclude the connection information
        events.loc[:, "component"] = [el.split("::")[0] for el in events.part_id.values]

        group_filter = []
        if frequency == "annual":
            group_filter.extend(["year"])
        elif frequency == "monthly":
            group_filter.extend(["month"])
        elif frequency == "month-year":
            group_filter.extend(["year", "month"])

        group_filter.append("component")
        cost_cols = ["total_cost"]
        if by_category:
            cost_cols[0:0] = [
                self._materials_cost,
                self._labor_cost,
                self._equipment_cost,
            ]

        if by_action:
            repair_map = {
                val: "repair" for val in ("repair request", "repair", "repair_complete")
            }
            maintenance_map = {
                val: "maintenance"
                for val in (
                    "maintenance request",
                    "maintenance",
                    "maintenance_complete",
                )
            }
            delay_map = {"delay": "delay"}
            action_map = {**repair_map, **maintenance_map, **delay_map}
            events.action = events.action.map(action_map)
            group_filter.append("action")

        month_year = frequency == "month-year"
        zeros = np.zeros(len(cost_cols)).tolist()
        costs = (
            events[group_filter + cost_cols].groupby(group_filter).sum().reset_index()
        )
        if not by_action:
            costs.loc[:, "action"] = np.zeros(costs.shape[0])
            cols = costs.columns.to_list()
            _ix = cols.index("component") + 1
            cols[_ix:_ix] = ["action"]
            cols.pop(-1)
            costs = costs.loc[:, cols]

        comparison_values: (
            product[tuple[Any, Any]]
            | product[tuple[Any, Any, Any]]
            | product[tuple[Any, Any, Any, Any]]
        )
        if frequency in ("annual", "month-year"):
            years = costs.year.unique()
            components = costs.component.unique()
            actions = costs.action.unique()
            comparison_values = product(years, components, actions)
            if month_year:
                months = costs.month.unique()
                comparison_values = product(years, months, components, actions)

            for _year, *_month, _component, _action in comparison_values:
                row_filter = costs.year.values == _year
                row_filter &= costs.component.values == _component
                row_filter &= costs.action.values == _action
                row = [_year, _component, _action] + zeros
                if month_year:
                    _month = _month[0]
                    row_filter &= costs.month.values == _month
                    row = [_year, _month, _component, _action] + zeros

                if costs.loc[row_filter].size > 0:
                    continue
                costs.loc[costs.shape[0]] = row
        elif frequency == "monthly":
            months = costs.month.unique()
            components = costs.component.unique()
            actions = costs.action.unique()
            comparison_values = product(months, actions, components)
            for _month, _action, _component in comparison_values:
                row_filter = costs.month.values == _month
                row_filter &= costs.component.values == _component
                row_filter &= costs.action.values == _action
                row = [_month, _component, _action] + zeros
                if costs.loc[row_filter].size > 0:
                    continue
                costs.loc[costs.shape[0]] = row
        elif frequency == "project":
            components = costs.component.unique()
            actions = costs.action.unique()
            comparison_values = product(actions, components)
            for _action, _component in comparison_values:
                row_filter = costs.component.values == _component
                row_filter &= costs.action.values == _action
                row = [_component, _action] + zeros
                if costs.loc[row_filter].size > 0:
                    continue
                costs.loc[costs.shape[0]] = row
        sort_cols = group_filter + cost_cols
        if group_filter != []:
            costs = costs.sort_values(group_filter)
        if sort_cols != []:
            costs = costs.loc[:, sort_cols]
        costs = costs.reset_index(drop=True)
        return costs if group_filter == [] else costs.set_index(group_filter)

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

        if frequency == "project":
            return pd.DataFrame([port_fee.sum(axis=0).loc[column]], columns=[column])
        elif frequency == "annual":
            return port_fee[["year"] + [column]].groupby(["year"]).sum()
        elif frequency == "monthly":
            return port_fee[["month"] + [column]].groupby(["month"]).sum()
        elif frequency == "month-year":
            return (
                port_fee[["year", "month"] + [column]].groupby(["year", "month"]).sum()
            )

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

        if frequency == "project":
            costs = pd.DataFrame(costs.reset_index(drop=True).sum()).T
        elif frequency == "annual":
            costs = costs.reset_index().groupby("year").sum().drop(columns=["month"])
        elif frequency == "monthly":
            costs = costs.reset_index().groupby("month").sum().drop(columns=["year"])

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
        if frequency == "project":
            materials = pd.DataFrame(materials.loc[:, ["materials_cost"]].sum()).T
        else:
            if frequency == "annual":
                group_col = ["year"]
            elif frequency == "monthly":
                group_col = ["month"]
            elif frequency == "month-year":
                group_col = ["year", "month"]
            materials = (
                materials[group_col + ["materials_cost"]].groupby(group_col).sum()
            )

        # Port fees will produce an 1x1 dataframe if values aren't present, so recreate
        # it with the appropriate dimension
        port_fees = self.port_fees(frequency=frequency)
        if frequency != "project" and port_fees.shape == (1, 1):
            port_fees = pd.DataFrame([], columns=["port_fees"], index=materials.index)
            port_fees = port_fees.fillna(0)

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

    def process_times(self) -> pd.DataFrame:
        """Calculates the time, in hours, to complete a repair/maintenance request, on
        both a request to completion basis, and the actual time to complete the repair.

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
        events_valid = self.events.loc[self.events.request_id != "na"]

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
                ["request_id", "reason"]
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
        timing = timing.join(reason_df[["reason"]]).rename(
            columns={"reason": "category"}
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
        return timing.groupby("category").sum().sort_index()

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
            Returns either a float for whole project-level costs or a pandas
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
            If ``by_turbine`` is not one of ``True`` or ``False``.
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

        if frequency == "annual":
            group_cols = ["year"]
        elif frequency == "monthly":
            group_cols = ["month"]
        elif frequency == "month-year":
            group_cols = ["year", "month"]

        col_filter = ["windfarm"]
        if by_turbine:
            col_filter.extend(self.turbine_id)

        if frequency == "project":
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
        return (
            self.production[group_cols + col_filter].groupby(by=group_cols).sum()
            / divisor
        )

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
        if frequency == "project":
            return pd.DataFrame(npv.reset_index().sum()).T[["NPV"]]
        elif frequency == "annual":
            return npv.reset_index().groupby("year").sum()[["NPV"]]
        elif frequency == "monthly":
            return npv.reset_index().groupby("month").sum()[["NPV"]]
        return npv[["NPV"]]
