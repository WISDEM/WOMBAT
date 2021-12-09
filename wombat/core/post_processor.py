"""The postprocessing metric computation."""

import os  # type: ignore
import numpy as np  # type: ignore
import PySAM
import pandas as pd  # type: ignore
import PySAM.PySSC as pssc  # type: ignore
import PySAM.Singleowner as pysam_singleowner_financial_model  # type: ignore
from copy import deepcopy  # type: ignore
from typing import List, Tuple, Union  # type: ignore
from pathlib import Path  # type: ignore
from functools import partial  # type: ignore
from itertools import product  # type: ignore

from wombat.core import FixedCosts
from wombat.core.library import load_yaml


def _calculate_time_availability(
    availability: np.ndarray, by_turbine=False
) -> Union[float, np.ndarray]:
    """Calculates the availability ratio of the whole timeseries or the whole timeseries, by turbine.

    Parameters
    ----------
    availability : np.ndarray
        Timeseries array of operating ratios.
    by_turbine : bool, optional
        If True, calculates the availability rate of each column, otherwise across the whole array, by default False.

    Returns
    -------
    Union[float, np.ndarray]
        Availability ratio across the whole timeseries, or broken out by column (turbine).
    """
    availability = availability > 0
    if by_turbine:
        return availability.sum(axis=0) / availability.shape[0]
    return availability.sum() / availability.size


def _process_single(
    events: pd.DataFrame, request_filter: np.ndarray
) -> Tuple[str, float, float, float, int]:
    """Computes the timing values for a single ``request_id``.

    Parameters
    ----------
    events : pd.DataFrame
        The NaN-filtered events ``pd.DataFrame``.
    request_filter : np.ndarray
        The indicies to include for the calculation of the timings.

    Returns
    -------
    Tuple[str, float, float, float, int]
        The timing values. See ``process_times``.
    """
    request = events.iloc[request_filter]
    downtime = request[request.system_operating_level < 1]
    vals = (
        request.reason[0],
        request.env_time.max() - request.env_time.min(),  # total time
        request.duration.sum(),  # actual process time
        downtime.env_time.max()
        - downtime.env_time.min(),  # downtime (duration of operations < 1)
        1,  # N processes
    )
    return vals


class Metrics:
    """The metric computation class that will store the logged outputs and compile results."""

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
        data_dir: Union[str, Path],
        events: Union[str, pd.DataFrame],
        operations: Union[str, pd.DataFrame],
        potential: Union[str, pd.DataFrame],
        production: Union[str, pd.DataFrame],
        inflation_rate: float,
        project_capacity: float,
        turbine_capacities: List[float],
        fixed_costs: str,
        substation_id: Union[str, List[str]],
        turbine_id: Union[str, List[str]],
        service_equipment_names: Union[str, List[str]],
        SAM_settings: Union[str, None] = None,
    ) -> None:
        """Initializes the Metrics class.

        Parameters
        ----------
        data_dir : Union[str, Path]
            This should be the same as was used for running the analysis.
        events : Union[str, pd.DataFrame]
            Either a pandas ``DataFrame`` or filename to be used to read the csv log data.
        operations : Union[str, pd.DataFrame]
            Either a pandas ``DataFrame`` or filename to be used to read the csv log data.
        potential : Union[str, pd.DataFrame]
            Either a pandas ``DataFrame`` or a filename to be used to read the csv
            potential power production data.
        production : Union[str, pd.DataFrame]
            Either a pandas ``DataFrame`` or a filename to be used to read the csv power
            production data.
        inflation_rate : float
            The inflation rate to be applied to all dollar amounts from the analysis
            starting year to ending year.
        project_capacity : float
            The project's rated capacity, in MW.
        turbine_capacities : Union[float, List[float]]
            The capacity of each individual turbine corresponding to ``turbine_id``, in kW.
        fixed_costs : str
            The filename of the project's fixed costs.
        substation_id : Union[str, List[str]]
            The substation id(s).
        turbine_id : Union[str, List[str]]
            The turbine id(s).
        service_equipment_names : Union[str, List[str]]
            The names of the servicing equipment, corresponding to
            ``ServiceEquipment.settings.name`` for each ``ServiceEquipment`` in the
            simulation.
        SAM_settings : Union[str, None]
            The SAM settings YAML file located in <data_dir>/windfarm/<SAM_settings>
            that should end in ".yaml". If no input is provided, then the model will
            raise a ``NotImplementedError`` when the SAM-powered metrics are attempted to
            be accessed.
        """
        self.data_dir = Path(data_dir)
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"{self.data_dir} does not exist")

        self.inflation_rate = 1 + inflation_rate
        self.project_capacity = project_capacity

        fixed_costs = load_yaml(self.data_dir / "windfarm", fixed_costs)
        self.fixed_costs = FixedCosts(**fixed_costs)  # type: ignore

        if isinstance(substation_id, str):
            substation_id = [substation_id]
        self.substation_id = substation_id

        if isinstance(turbine_id, str):
            turbine_id = [turbine_id]
        self.turbine_id = turbine_id

        if isinstance(service_equipment_names, str):
            service_equipment_names = [service_equipment_names]
        self.service_equipment_names = sorted(list(set(service_equipment_names)))

        if isinstance(turbine_capacities, (float, int)):
            turbine_capacities = [turbine_capacities]
        self.turbine_capacities = turbine_capacities

        if isinstance(events, str):
            events = self._read_data(events)
        self.events = self._apply_inflation_rate(self._tidy_data(events, kind="events"))

        if isinstance(operations, str):
            operations = self._read_data(operations)
        self.operations = self._tidy_data(operations, kind="operations")

        if isinstance(potential, str):
            potential = self._read_data(potential)
        self.potential = self._tidy_data(potential, kind="potential")

        if isinstance(production, str):
            production = self._read_data(production)
        self.production = self._tidy_data(production, kind="production")

        if SAM_settings is not None:
            SAM_settings = "SAM_Singleowner_defaults.yaml"
            self.sam_settings = load_yaml(self.data_dir / "windfarm", SAM_settings)
            self._setup_pysam()
        else:
            self.sam_settings = None
            self.financial_model = None

    def _tidy_data(self, data: pd.DataFrame, kind: str) -> pd.DataFrame:
        """Tidies the "raw" csv-converted data to be able to be used among the ``Metrics``
        class.

        Parameters
        ----------
        data : pd.DataFrame
            The freshly imported csv log data.
        kind : str
            The category of the input provided to ``data``. Should be one of:
             - "operations"
             - "events"
             - "potential"
             - "production"

        Returns
        -------
        pd.DataFrame
            A tidied data frame to be used for all the operations in this class.
        """
        if data.index.name != "datetime":
            try:
                data.datetime = pd.to_datetime(data.datetime)
            except AttributeError:
                data["datetime"] = pd.to_datetime(data.env_datetime)
            data.index = data.datetime
            data = data.drop(labels="datetime", axis=1)
        data.env_datetime = pd.to_datetime(data.env_datetime)
        data["year"] = data.env_datetime.dt.year
        data["month"] = data.env_datetime.dt.month
        data["day"] = data.env_datetime.dt.day
        if kind == "operations":
            data["windfarm"] = data[self.substation_id].mean(axis=1) * data[
                self.turbine_id
            ].mean(axis=1)
        elif kind in ("potential", "production"):
            data[self.turbine_id] = data[self.turbine_id].astype(float)
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
        data = pd.read_csv(self.data_dir / "outputs" / "logs" / fname)
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

    def _setup_pysam(self) -> None:
        """Creates and executes the PySAM model for financial metrics."""
        # Define the model and import the SAM settings file.
        self.financial_model = pysam_singleowner_financial_model.default(
            "WindPowerSingleOwner"
        )
        model_data = pssc.dict_to_ssc_table(self.sam_settings, "singleowner")
        self.financial_model = pysam_singleowner_financial_model.wrap(model_data)

        # Remove the leap year production
        leap_year_ix = self.production.index.month == 2
        leap_year_ix &= self.production.index.day == 29
        generation = self.production.loc[~leap_year_ix].windfarm.values

        # Create a years variable for later use with the PySAM outputs
        self.years = sorted(self.production.year.unique())

        # Let mypy know that I know what I'm doing
        assert isinstance(self.financial_model, PySAM.Singleowner.Singleowner)

        # Replace the coded generation with modeled generation
        self.financial_model.FinancialParameters.analysis_period = len(self.years)
        self.financial_model.SystemOutput.gen = generation

        # Reset the system capacity, in kW
        self.financial_model.FinancialParameters.system_capacity = (
            self.project_capacity * 1000
        )

        # Run the financial model
        self.financial_model.execute()

    def time_based_availability(  # type: ignore
        self, frequency: str, by: str
    ) -> Union[float, pd.DataFrame]:
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
        Union[float, pd.DataFrame]
            The time-based availability at the desired aggregation level.
        """
        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual", "monthly", "month-year"):
            raise ValueError(
                '``frequency`` must be one of "project", "annual", "monthly", or "month-year".'
            )

        by = by.lower().strip()
        if by not in ("windfarm", "turbine"):
            raise ValueError('``by`` must be one of "windfarm" or "turbine".')
        by_turbine = by == "turbine"

        operations = deepcopy(self.operations)
        substation = np.prod(operations[self.substation_id], axis=1).values.reshape(
            (-1, 1)
        )
        hourly = substation * operations.loc[:, self.turbine_id].values

        if frequency == "project":
            availability = _calculate_time_availability(hourly, by_turbine=by_turbine)
            if by == "windfarm":
                return availability
            availability = pd.DataFrame(
                availability.reshape(1, -1), columns=self.turbine_id  # type: ignore
            )
            return availability
        elif frequency == "annual":
            date_time = operations[["year"]]
            counts = operations.groupby(by="year").count()
            counts = counts[self.turbine_id] if by_turbine else counts[["windfarm"]]
            annual = [
                _calculate_time_availability(
                    hourly[date_time.year == year], by_turbine=by_turbine
                )
                for year in counts.index
            ]
            return pd.DataFrame(
                annual, index=counts.index, columns=counts.columns
            ).reset_index()
        elif frequency == "monthly":
            date_time = operations[["month"]]
            counts = operations.groupby(by="month").count()
            counts = counts[self.turbine_id] if by_turbine else counts[["windfarm"]]
            monthly = [
                _calculate_time_availability(
                    hourly[date_time.month == month], by_turbine=by_turbine
                )
                for month in counts.index
            ]
            return pd.DataFrame(
                monthly, index=counts.index, columns=counts.columns
            ).reset_index()
        elif frequency == "month-year":
            date_time = operations[["year", "month"]]
            counts = operations.groupby(by=["year", "month"]).count()
            counts = counts[self.turbine_id] if by_turbine else counts[["windfarm"]]
            month_year = [
                _calculate_time_availability(
                    hourly[(date_time.year == year) & (date_time.month == month)],
                    by_turbine=by_turbine,
                )
                for year, month in counts.index
            ]
            return pd.DataFrame(
                month_year, index=counts.index, columns=counts.columns
            ).reset_index()

    def production_based_availability(  # type: ignore
        self, frequency: str, by: str
    ) -> Union[float, pd.DataFrame]:
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
        Union[float, pd.DataFrame]
            The production-based availability at the desired aggregation level.
        """
        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual", "monthly", "month-year"):
            raise ValueError(
                '``frequency`` must be one of "project", "annual", "monthly", or "month-year".'
            )

        by = by.lower().strip()
        if by not in ("windfarm", "turbine"):
            raise ValueError('``by`` must be one of "windfarm" or "turbine".')
        by_turbine = by == "turbine"

        production = self.production.loc[:, self.turbine_id]
        potential = self.potential.loc[:, self.turbine_id]

        if frequency == "project":
            production = production.values
            potential = potential.values
            if (potential == 0).sum() > 0:
                potential[potential == 0] = 1
            if not by_turbine:
                return production.sum() / potential.sum()
            availability = pd.DataFrame(
                (production.sum(axis=0) / potential.sum(axis=0)).reshape(1, -1),
                columns=self.turbine_id,
            )
            return availability

        production[["year", "month"]] = [
            production.index.year.values.reshape(-1, 1),
            production.index.month.values.reshape(-1, 1),
        ]
        potential[["year", "month"]] = [
            potential.index.year.values.reshape(-1, 1),
            potential.index.month.values.reshape(-1, 1),
        ]

        if frequency == "annual":
            production = production.groupby("year").sum()[self.turbine_id]
            potential = potential.groupby("year").sum()[self.turbine_id]

        elif frequency == "monthly":
            production = production.groupby("month").sum()[self.turbine_id]
            potential = potential.groupby("month").sum()[self.turbine_id]

        elif frequency == "month-year":
            production = production.groupby(["year", "month"]).sum()[self.turbine_id]
            potential = potential.groupby(["year", "month"]).sum()[self.turbine_id]

        if (potential.values == 0).sum() > 0:
            potential.loc[potential.values == 0] = 1
        columns = self.turbine_id
        if not by_turbine:
            production = production.sum(axis=1)
            potential = potential.sum(axis=1)
            columns = [by]
        return pd.DataFrame(production / potential, columns=columns)

    def capacity_factor(  # type: ignore
        self, which: str, frequency: str, by: str
    ) -> Union[float, pd.DataFrame]:
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
        Union[float, pd.DataFrame]
            The capacity factor at the desired aggregation level.
        """
        which = which.lower().strip()
        if which not in ("net", "gross"):
            raise ValueError('``which`` must be one of "net" or "gross".')

        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual", "monthly", "month-year"):
            raise ValueError(
                '``frequency`` must be one of "project", "annual", "monthly", or "month-year".'
            )

        by = by.lower().strip()
        if by not in ("windfarm", "turbine"):
            raise ValueError('``by`` must be one of "windfarm" or "turbine".')
        by_turbine = by == "turbine"
        capacity = (
            np.array(self.turbine_capacities) if by_turbine else self.project_capacity
        )

        production = self.production if which == "net" else self.potential
        production = production.loc[:, self.turbine_id]

        if frequency == "project":
            potential = production.shape[0]
            if not by_turbine:
                production = production.values.sum() / 1000  # convert to MWh
                return production / (potential * capacity)
            potential = potential * capacity / 1000
            cf = pd.DataFrame((production.sum(axis=0) / 1000 / potential)).T
            return cf

        production[["year", "month"]] = [
            production.index.year.values.reshape(-1, 1),
            production.index.month.values.reshape(-1, 1),
        ]

        if frequency == "annual":
            potential = production.groupby("year").count()[self.turbine_id]
            production = production.groupby("year").sum()[self.turbine_id]

        elif frequency == "monthly":
            potential = production.groupby("month").count()[self.turbine_id]
            production = production.groupby("month").sum()[self.turbine_id]

        elif frequency == "month-year":
            potential = production.groupby(["year", "month"]).count()[self.turbine_id]
            production = production.groupby(["year", "month"]).sum()[self.turbine_id]

        if by_turbine:
            columns = self.turbine_id
            potential = potential.iloc[:, 0].values.reshape(-1, 1) * (
                capacity / 1000
            ).reshape(1, -1)
        else:
            production = production.sum(axis=1)
            potential = potential.iloc[:, 0] * capacity
            columns = [by]
        return pd.DataFrame(production / 1000 / potential, columns=columns)

    def task_completion_rate(
        self, which: str, frequency: str
    ) -> Union[float, pd.DataFrame]:
        """Calculates the task completion rate over a project's lifetime as a single value,
        annual average, or monthly average for the whole windfarm or by turbine.

        .. note:: This currently assumes that if there are multiple substations, that
          the turbines are all connected to multiple.

        Parameters
        ----------
        which : str
            One of "scheduled", "unscheduled", or "both".
        frequency : str
            One of "project", "annual", "monthly", or "month-year".

        Returns
        -------
        Union[float, pd.DataFrame]
            The task completion rate at the desired aggregation level.
        """
        which = which.lower().strip()
        if which not in ("scheduled", "unscheduled", "both"):
            raise ValueError(
                '``which`` must be one of "scheduled", "unscheduled", or "both".'
            )

        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual", "monthly", "month-year"):
            raise ValueError(
                '``frequency`` must be one of "project", "annual", "monthly", or "month-year".'
            )

        if which == "scheduled":
            task_filter = ["maintenance"]
        elif which == "unscheduled":
            task_filter = ["repair"]
        else:
            task_filter = ["maintenance", "repair"]

        cols = ["env_datetime", "request_id"]
        completion_filter = [f"{el} complete" for el in task_filter]
        request_filter = [f"{el} request" for el in task_filter]
        requests = self.events.loc[
            self.events.action.isin(request_filter), cols
        ].reset_index(drop=True)
        completions = self.events.loc[
            self.events.action.isin(completion_filter), cols
        ].reset_index(drop=True)

        if frequency == "project":
            return completions.shape[0] / requests.shape[0]

        requests[["year", "month"]] = [
            requests.env_datetime.dt.year.values.reshape(-1, 1),
            requests.env_datetime.dt.month.values.reshape(-1, 1),
        ]

        completions[["year", "month"]] = [
            completions.env_datetime.dt.year.values.reshape(-1, 1),
            completions.env_datetime.dt.month.values.reshape(-1, 1),
        ]

        if frequency == "annual":
            group_filter = ["year"]
            indices = self.operations.year.unique()
        elif frequency == "monthly":
            group_filter = ["month"]
            indices = self.operations.month.unique()
        elif frequency == "month-year":
            group_filter = ["year", "month"]
            indices = list(
                product(self.operations.year.unique(), self.operations.month.unique())
            )

        requests = requests.groupby(group_filter).count()["request_id"]
        requests.loc[requests == 0] = 1
        completions = completions.groupby(group_filter).count()["request_id"]

        missing = [ix for ix in indices if ix not in requests]
        requests = requests.append(pd.Series(np.ones(len(missing)), index=missing))
        requests = requests.sort_index()

        missing = [ix for ix in indices if ix not in completions]
        completions = completions.append(
            pd.Series(np.zeros(len(missing)), index=missing)
        )
        completions = completions.sort_index()

        completion_rate = pd.DataFrame(completions / requests).reset_index()
        return completion_rate.rename(
            columns={
                "request_id": "Completion Rate",
                0: "Completion Rate",
                "index": "year",
            }
        )

    def equipment_costs(
        self, frequency: str, by_equipment: bool = False
    ) -> Union[float, pd.DataFrame]:
        """Calculates the equipment costs for the simulation at a project, annual, or
        monthly level with (or without) respect to equipment utilized in the simulation.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by_equipment : bool, optional
            Indicates whether the values are with resepect to the equipment utilized
            (True) or not (False), by default False.

        Returns
        -------
        Union[float, pd.DataFrame]
            Returns either a float for whole project-level costs or a pandas ``DataFrame``
            with columns:
                - year (if appropriate for frequency)
                - month (if appropriate for frequency)
                - then any equipment names as they appear in the logs

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or "month-year".
        ValueError
            If ``by_equipment`` is not one of ``True`` or ``False``.
        """
        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual", "monthly", "month-year"):
            raise ValueError(
                '``frequency`` must be one of "project", "annual", or "monthly".'
            )

        if not isinstance(by_equipment, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")

        if frequency == "annual":
            col_filter = ["year"]
        elif frequency == "monthly":
            col_filter = ["month"]
        elif frequency == "month-year":
            col_filter = ["year", "month"]

        if by_equipment:
            if frequency == "project":
                costs = (
                    self.events[self.events[self._equipment_cost] > 0]
                    .groupby(["agent"])
                    .sum()[[self._equipment_cost]]
                    .fillna(0)
                    .reset_index(level=0)
                )
                costs = costs.fillna(costs.max(axis=0)).T
                costs = (
                    costs.rename(columns=costs.iloc[0])
                    .drop(index="agent")
                    .reset_index(drop=True)
                )
                return costs

            col_filter = ["agent"] + col_filter
            costs = (
                self.events[self.events[self._equipment_cost] > 0]
                .groupby(col_filter)
                .sum()[[self._equipment_cost]]
                .reset_index(level=0)
            )
            costs = pd.concat(
                [
                    costs[costs.agent == eq][[self._equipment_cost]].rename(
                        columns={self._equipment_cost: eq}
                    )
                    for eq in costs.agent.unique()
                ],
                axis=1,
            )
            return costs.fillna(value=0).reset_index()

        if frequency == "project":
            return self.events[self._equipment_cost].sum()

        costs = self.events.groupby(col_filter).sum()[[self._equipment_cost]]
        return costs.fillna(0).reset_index()

    def service_equipment_utilization(self, frequency: str) -> pd.DataFrame:
        """Calculates the utilization rate for each of the service equipment in the
        simulation.

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
        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual"):
            raise ValueError('``frequency`` must be one of "project" or "annual".')

        no_requests = []
        total_days = []
        no_request_filter = self.events.reason == "no requests"
        return_filter = self.events.action == "delay"
        return_filter &= self.events.reason == "work is complete"
        return_filter &= self.events.additional == "will return next year"
        for name in self.service_equipment_names:
            equipment_filter = self.events.agent == name
            _events = self.events[equipment_filter & no_request_filter]
            _events = _events.groupby("year").count()[["agent"]]
            no_requests.append(_events.rename(columns={"agent": name}))

            ix_filter = equipment_filter & ~return_filter
            total = self.events[ix_filter].groupby(["year", "month", "day"]).size()
            total = total.reset_index().groupby("year").count()[["day"]]
            total_days.append(total.rename(columns={"day": name}))

        if len(self.service_equipment_names) > 1:
            no_requests_df = no_requests[0].join(no_requests[1:]).fillna(0)
            total_df = total_days[0].join(total_days[1:]).fillna(1)
        else:
            no_requests_df = pd.DataFrame(no_requests[0])
            total_df = pd.DataFrame(total_days[0])

        for year in self.events.year.unique():
            if year not in no_requests_df.index:
                missing = pd.DataFrame(
                    np.zeros((1, no_requests_df.shape[1])),
                    index=[year],
                    columns=no_requests_df.columns,
                )
                no_requests_df = no_requests_df.append(missing).sort_index()
            if year not in total_df.index:
                missing = pd.DataFrame(
                    np.ones((1, total_df.shape[1])),
                    index=[year],
                    columns=no_requests_df.columns,
                )
                total_df = total_df.append(missing).sort_index()

        if frequency == "project":
            no_requests_df = no_requests_df.reset_index().sum()[
                self.service_equipment_names
            ]
            total_df = total_df.reset_index().sum()[self.service_equipment_names]
            return pd.DataFrame((total_df - no_requests_df) / total_df).T
        return (total_df - no_requests_df) / total_df

    def labor_costs(
        self, frequency: str, by_type: bool = False
    ) -> Union[float, pd.DataFrame]:
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
        Union[float, pd.DataFrame]
            Returns either a float for whole project-level costs or a pandas ``DataFrame``
            with columns:

             - year (if appropriate for frequency)
             - month (if appropriate for frequency)
             - total_labor_cost
             - hourly_labor_cost (if broken out)
             - salary_labor_cost (if broken out)

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or "month-year".
        ValueError
            If ``by_type`` is not one of ``True`` or ``False``.
        """
        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual", "monthly", "month-year"):
            raise ValueError(
                '``frequency`` must be one of "project", "annual", "monthly", or "month-year".'
            )

        if not isinstance(by_type, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")

        labor_cols = [self._hourly_cost, self._salary_cost, self._labor_cost]
        if frequency == "project":
            costs = pd.DataFrame(
                self.events[labor_cols].sum(axis=0).values.reshape(1, -1),
                columns=labor_cols,
            )
            if not by_type:
                return costs[self._labor_cost].values[0]
            return costs

        if frequency == "annual":
            group_filter = ["year"]
        elif frequency == "monthly":
            group_filter = ["month"]
        elif frequency == "month-year":
            group_filter = ["year", "month"]

        costs = (
            self.events.groupby(group_filter)
            .sum()[labor_cols]
            .reset_index()
            .fillna(value=0)
        )
        if not by_type:
            return costs[group_filter + [self._labor_cost]]
        return costs

    def equipment_labor_cost_breakdowns(
        self, frequency: str, by_category: bool = False
    ) -> pd.DataFrame:
        """Calculates the producitivty cost breakdowns for the simulation at a project, annual, or
        monthly level that can be broken out to include the equipment and labor components.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by_category : bool, optional
            Indicates whether to include the equipment and labor categories (True) or
            not (False), by default False.

        .. note:: Does not produce a value if there is no cost associated with a "reason".

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

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or "month-year".
        ValueError
            If ``by_category`` is not one of ``True`` or ``False``.
        """
        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual", "monthly", "month-year"):
            raise ValueError(
                '``frequency`` must be one of "project", "annual", or "monthly".'
            )
        if not isinstance(by_category, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")

        group_filter = ["action", "reason", "additional"]
        if frequency in ("annual", "month-year"):
            group_filter.insert(0, "year")
        elif frequency == "monthly":
            group_filter.insert(0, "month")
        if frequency == "month-year":
            group_filter.insert(1, "month")

        equipment = self.events[self.events[self._equipment_cost] > 0].agent.unique()
        costs = (
            self.events[
                self.events.agent.isin(equipment)
                & self.events.action.isin(
                    ("delay", "repair", "maintenance", "mobilization")
                )
                & ~self.events.additional.isin(["work is complete"])
            ]
            .groupby(group_filter)
            .sum()[self._cost_columns]
            .reset_index()
        )
        costs["display_reason"] = [""] * costs.shape[0]
        group_filter.append("display_reason")

        non_shift_hours = (
            "not in working hours",
            "work shift has ended; waiting for next shift to start",
            "no more return visits will be made",
            "will return next year",
        )
        costs.loc[
            (costs.action == "delay") & (costs.additional.isin(non_shift_hours)),
            "display_reason",
        ] = "Not in Shift"
        costs.loc[costs.action == "repair", "display_reason"] = "Repair"
        costs.loc[costs.action == "maintenance", "display_reason"] = "Maintenance"
        costs.loc[costs.action == "mobilization", "display_reason"] = "Mobilization"
        costs.loc[
            costs.additional == "weather delay", "display_reason"
        ] = "Weather Delay"
        costs.loc[costs.reason == "no requests", "display_reason"] = "No Requests"

        costs.reason = costs.display_reason
        group_filter.pop(group_filter.index("action"))
        group_filter.pop(group_filter.index("display_reason"))
        group_filter.pop(group_filter.index("additional"))

        drop_columns = [self._materials_cost]
        if not by_category:
            drop_columns.extend(
                [
                    self._hourly_cost,
                    self._salary_cost,
                    self._labor_cost,
                    self._equipment_cost,
                ]
            )
        costs = costs.drop(columns=drop_columns)
        costs = costs.groupby(group_filter).sum().reset_index()

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
            "Mobilization",
            "Weather Delay",
            "No Requests",
            "Not in Shift",
        ]
        costs.reason = pd.Categorical(costs.reason, new_sort)
        if frequency == "project":
            return costs.sort_values(by="reason").reset_index(drop=True)
        if frequency == "annual":
            return costs.sort_values(by=["year", "reason"]).reset_index(drop=True)
        if frequency == "monthly":
            return costs.sort_values(by=["month", "reason"]).reset_index(drop=True)
        return costs.sort_values(by=["year", "month", "reason"]).reset_index(drop=True)

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
        Union[float, pd.DataFrame]
            Returns either a float for whole project-level costs or a pandas ``DataFrame``
            with columns:
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
            If ``frequency`` is not one of "project", "annual", "monthly", or "month-year".
        ValueError
            If ``by_category`` is not one of ``True`` or ``False``.
        ValueError
            If ``by_action`` is not one of ``True`` or ``False``.
        """
        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual", "monthly", "month-year"):
            raise ValueError(
                '``frequency`` must be one of "project", "annual", or "monthly".'
            )
        if not isinstance(by_category, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")
        if not isinstance(by_action, bool):
            raise ValueError("``by_equipment`` must be one of ``True`` or ``False``")

        events = deepcopy(self.events)
        events = events[~events.part_id.isna()]

        # Need to simplify the cable identifiers to not include the connection information
        events["component"] = [el.split("::")[0] for el in events.part_id.values]

        cost_cols = ["total_cost"]
        if by_category:
            cost_cols[0:0] = ["materials_cost", "total_labor_cost", "equipment_cost"]

        group_filter = []
        if frequency == "annual":
            group_filter = ["year"]
        elif frequency == "monthly":
            group_filter = ["month"]
        elif frequency == "month-year":
            group_filter = ["year", "month"]
        group_filter.append("component")

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
        costs = events.groupby(group_filter).sum()[cost_cols].reset_index()
        if not by_action:
            costs["action"] = np.zeros(costs.shape[0])
            cols = costs.columns.to_list()
            _ix = cols.index("component") + 1
            cols[_ix:_ix] = ["action"]
            cols.pop(-1)
            costs = costs[cols]
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
        costs = costs.sort_values(group_filter)[group_filter + cost_cols]
        return costs.reset_index(drop=True)

    def project_fixed_costs(self, frequency: str, resolution: str) -> pd.DataFrame:
        """Calculates the fixed costs of a project at the project and annual frequencies
        at a given cost breakdown resolution.

        Parameters
        ----------
        frequency : str
            One of "project" or "annual".
        resolution : str
            One of "low", "medium", or "high", where the values correspond to:

             - low: ``FixedCosts.resolution["low"]``, corresponding to the itemized costs.
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
        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual"):
            raise ValueError('``frequency`` must be one of "project" or "annual".')

        resolution = resolution.lower().strip()
        if resolution not in ("low", "medium", "high"):
            raise ValueError(
                '``resolution`` must be one of "low", "medium", or "high".'
            )

        keys = self.fixed_costs.resolution[resolution]
        vals = [[getattr(self.fixed_costs, key) for key in keys]]
        costs = pd.DataFrame(vals, columns=keys)

        years = self.events.year.unique()
        years.sort()
        adjusted_inflation = deepcopy(self.inflation_rate)
        vals = np.array(vals[0]) * self.project_capacity * 1000
        for ix, year in enumerate(years):
            hours = self.operations[self.operations.year == year].shape[0]
            costs.loc[ix] = vals * min(1, hours / 8760.0)
            if ix == 0:
                continue
            costs.loc[ix] *= adjusted_inflation
            adjusted_inflation *= self.inflation_rate

        if frequency == "project":
            return pd.DataFrame(costs.sum(axis=0).values.reshape(1, -1), columns=keys)

        costs["year"] = years
        col_resort = ["year"] + keys
        costs = costs[col_resort]
        return costs

    def process_times(self) -> pd.DataFrame:
        """Calculates the time, in hours, to complete a repair/maintenance request, on both a
        request to completion basis, and the actual time to complete the repair.

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
        events = self.events.loc[self.events.request_id != "na"]
        requests = events.request_id.values
        unique = events.request_id.unique()
        process_single = partial(_process_single, events)
        timing = [process_single(np.where(requests == rid)[0]) for rid in unique]
        df = pd.DataFrame(
            timing,
            columns=["category", "time_to_completion", "process_time", "downtime", "N"],
            dtype=float,
        )
        df = df.groupby("category").sum()
        df = df.sort_index()
        return df

    def power_production(
        self, frequency: str, by_turbine: bool = False
    ) -> Union[float, pd.DataFrame]:
        """Calculates the power production for the simulation at a project, annual, or
        monthly level that can be broken out by hourly and salary labor costs.

        Parameters
        ----------
        frequency : str
            One of "project", "annual", "monthly", or "month-year".
        by_turbine : bool, optional
            Indicates whether the values are with resepect to the individual turbines
            (True) or the windfarm (False), by default False.

        Returns
        -------
        Union[float, pd.DataFrame]
            Returns either a float for whole project-level costs or a pandas ``DataFrame``
            with columns:

             - year (if appropriate for frequency)
             - month (if appropriate for frequency)
             - total_power_production
             - <turbine_id>_power_production (if broken out)

        Raises
        ------
        ValueError
            If ``frequency`` is not one of "project", "annual", "monthly", or "month-year".
        ValueError
            If ``by_turbine`` is not one of ``True`` or ``False``.
        """
        frequency = frequency.lower().strip()
        if frequency not in ("project", "annual", "monthly", "month-year"):
            raise ValueError(
                '``frequency`` must be one of "project", "annual", or "monthly".'
            )

        if not isinstance(by_turbine, bool):
            raise ValueError("``by_turbine`` must be one of ``True`` or ``False``")

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
            production = self.production.sum(axis=0)[col_filter]
            production = pd.DataFrame(
                production.values.reshape(1, -1),
                columns=col_filter,
                index=["Project Energy Production (kWh)"],
            )
            return production
        return self.production.groupby(by=group_cols).sum()[col_filter]

    # Windfarm Financials

    def pysam_npv(self) -> Union[float, pd.DataFrame]:
        """Returns the project-level after-tax net present values (NPV).

        See here for more: https://nrel-pysam.readthedocs.io/en/master/modules/Singleowner.html#PySAM.Singleowner.Singleowner.Outputs.cf_project_return_aftertax_npv

        Raises
        ------
        NotImplementedError: Raised if a PySAM input file is not provided to run the model.

        Returns
        -------
        float
            Final, project-level NPV, in $.
        """
        if self.financial_model is None:
            raise NotImplementedError(
                "No SAM inputs were provided, and 'pysam_npv()' cannot be calculated!"
            )
        npv = self.financial_model.Outputs.cf_project_return_aftertax_npv
        npv = npv[len(self.years)]
        return npv

    def pysam_lcoe_real(self) -> float:
        """Returns the real levelized cost of energy (LCOE) from PySAM.

        See here for more: https://nrel-pysam.readthedocs.io/en/master/modules/Singleowner.html#PySAM.Singleowner.Singleowner.Outputs.lcoe_real

        Raises
        ------
        NotImplementedError: Raised if a PySAM input file is not provided to run the model.

        Returns
        -------
        float
            Real LCOE, in $/kW.
        """
        if self.financial_model is None:
            raise NotImplementedError(
                "No SAM inputs were provided, and 'pysam_lcoe_real()' cannot be calculated!"
            )
        return self.financial_model.Outputs.lcoe_real / 100.0

    def pysam_lcoe_nominal(self) -> float:
        """Returns the nominal levelized cost of energy (LCOE) from PySAM.

        See here for more: https://nrel-pysam.readthedocs.io/en/master/modules/Singleowner.html#PySAM.Singleowner.Singleowner.Outputs.lcoe_nom

        Raises
        ------
        NotImplementedError: Raised if a PySAM input file is not provided to run the model.

        Returns
        -------
        float
            Nominal LCOE, in $/kW.
        """
        if self.financial_model is None:
            raise NotImplementedError(
                "No SAM inputs were provided, and 'pysam_lcoe_nominal()' cannot be calculated!"
            )
        return self.financial_model.Outputs.lcoe_nom / 100.0

    def pysam_irr(self) -> float:
        """Returns the project-level after-tax internal return rate (IRR).

        See here for more: https://nrel-pysam.readthedocs.io/en/master/modules/Singleowner.html#PySAM.Singleowner.Singleowner.Outputs.cf_project_return_aftertax_irr

        Raises
        ------
        NotImplementedError: Raised if a PySAM input file is not provided to run the model.

        Returns
        -------
        pd.DataFrame
            Annual after-tax IRR value, in %.
        """
        if self.financial_model is None:
            raise NotImplementedError(
                "No SAM inputs were provided, and 'pysam_irr()' cannot be calculated!"
            )
        irr = self.financial_model.Outputs.cf_project_return_aftertax_irr
        irr = irr[len(self.years)]
        return irr

    def pysam_all_outputs(self) -> pd.DataFrame:
        """Returns all the possible PySAM outputs that are included in this module as
        columns in the following order:

         - NPV
         - Nominal LCOE
         - Real LOCE
         - IRR

        Raises
        ------
        NotImplementedError: Raised if a PySAM input file is not provided to run the model.

        Returns
        -------
        pd.DataFrame
            Project financial values values.
        """
        if self.financial_model is None:
            raise NotImplementedError(
                "No SAM inputs were provided, and 'pysam_all_outputs()' cannot be calculated!"
            )
        financials = [
            self.pysam_npv(),
            self.pysam_lcoe_nominal(),
            self.pysam_lcoe_real(),
            self.pysam_irr(),
        ]
        descriptions = [
            "After Tax NPV ($)",
            "Nominal LCOE ($/kW)",
            "Real LCOE ($/kW)",
            "After Tax IRR (%)",
        ]
        financials = pd.DataFrame(
            financials, index=descriptions, dtype=float, columns=["Value"]
        )
        financials.index.name = "Metric"  # type: ignore
        return financials
