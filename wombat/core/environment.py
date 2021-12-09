"""Provides the O&M Enviroment class; a subclass of simpy.Environment."""


import os  # type: ignore
import math  # type: ignore
import numpy as np  # type: ignore
import simpy  # type: ignore
import pandas as pd  # type: ignore
import logging  # type: ignore
from typing import Tuple, Union, Optional  # type: ignore
from pathlib import Path  # type: ignore
from datetime import datetime, timedelta  # type: ignore
from simpy.events import Event  # type: ignore
from pandas.core.indexes.datetimes import DatetimeIndex

import wombat  # pylint: disable=W0611
from wombat.utilities import (
    setup_logger,
    hours_until_future_hour,
    format_events_log_message,
)


class WombatEnvironment(simpy.Environment):
    """The primary mechanism for powering an O&M simulation. This object has insight into
    all other simulation objects, and controls the timing, date/time stamps, and weather
    conditions.
    """

    def __init__(
        self,
        data_dir: Union[Path, str],
        weather_file: str,
        workday_start: int,
        workday_end: int,
        simulation_name: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> None:
        """Initialization.

        Parameters
        ----------
        simpy : [type]
            [description]
        data_dir : str
            Directory where the inputs are stored and where to save outputs.
        weather_file : str
            Name of the weather file. Should be contained within ``data_dir``/weather/.
        workday_start : int
            Starting time for the repair crew, in 24 hour local time. This can be
            overridden by an ``ServiceEquipmentData`` object that operates outside of the "typical"
            working hours.
        workday_end : int
            Ending time for the repair crew, in 24 hour local time. This can be
            overridden by an ``ServiceEquipmentData`` object that operates outside of the "typical"
            working hours.
        simulation_name : Optional[str], optional
            Name of the simulation; will be used for naming the log file, by default ``None``.
            If ``None``, then the current time will be used. Will always save to
            ``data_dir``/outputs/logs/``simulation_name``.log.
            ... note: spaces (" ") will be replaced with underscores ("_"), for example:
            "my example analysis" becomes "my_example_analysis".
        start_year : Optional[int], optional
            Custom starting year for the weather profile, by default None. If ``None`` or
            less than the first year of the weather profile, this will be ignored.
        end_year : Optional[int], optional
            Custom ending year for the weather profile, by default None. If ``None`` or
            greater than the last year of the weather profile, this will be ignored.

        Raises
        ------
        FileNotFoundError
            Raised if ``data_dir`` cannot be found.
        """
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"{self.data_dir} does not exist")

        self.workday_start = workday_start
        self.workday_end = workday_end
        if not 0 <= self.workday_start < 24:
            raise ValueError("workday_start must be a valid 24hr time before midnight!")
        if self.workday_end > 24:
            raise ValueError("Work shifts must end before midnight")
        if 0 < self.workday_end <= self.workday_start:
            raise ValueError(
                "Work shifts must end after they start ({self.workday_start}hrs)."
            )

        self.weather = self._weather_setup(weather_file, start_year, end_year)
        self.max_run_time = self.weather.shape[0]
        self._create_endpoints()
        self.shift_length = self.workday_end - self.workday_start

        self.simulation_name = simulation_name
        self._logging_setup()

    def run(self, until: Optional[Union[int, float, Event]] = None):
        """Extends the ``simpy.Environment.run`` method to change the default behavior if
        no argument is passed to ``until``, which will now run a simulation until the end
        of the weather profile is reached.

        Parameters
        ----------
        until : Optional[Union[int, float, Event]], optional
            When to stop the simulation, by default None. See documentation on
            ``simpy.Environment.run`` for more details.
        """
        if until is None:
            until = self.max_run_time
        elif until < self.max_run_time:
            self.max_run_time = until
        super().run(until=until)

    def _logging_setup(self) -> None:
        """Completes the setup for logging data."""
        if self.simulation_name is None:
            simulation = "wombat"
        else:
            simulation = self.simulation_name.replace(" ", "_")
        dt_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

        self.events_log_fname = f"{dt_stamp}_{simulation}_events.log"
        self.operations_log_fname = f"{dt_stamp}_{simulation}_operations.log"
        self.power_potential_fname = f"{dt_stamp}_{simulation}_power_potential.csv"
        self.power_production_fname = f"{dt_stamp}_{simulation}_power_production.csv"

        self.events_log_fname = str(
            self.data_dir / "outputs" / "logs" / self.events_log_fname
        )
        self.operations_log_fname = str(
            self.data_dir / "outputs" / "logs" / self.operations_log_fname
        )
        self.power_potential_fname = str(
            self.data_dir / "outputs" / "logs" / self.power_potential_fname
        )
        self.power_production_fname = str(
            self.data_dir / "outputs" / "logs" / self.power_production_fname
        )

        _dir = self.data_dir / "outputs" / "logs"
        if not os.path.isdir(str(_dir)):
            os.makedirs(_dir)

        # logging.basicConfig(
        #     format="%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s",
        #     filename=self.events_log_fname,
        #     filemode="w",
        #     level=logging.DEBUG,
        # )
        # self._events_logger = logging.getLogger(__name__)

        setup_logger("events_log", self.events_log_fname)
        setup_logger("operations_log", self.operations_log_fname)
        self._events_logger = logging.getLogger("events_log")
        self._operations_logger = logging.getLogger("operations_log")

    @property
    def simulation_time(self) -> datetime:
        """Current time within the simulation ("datetime" column within weather)."""
        now = self.now
        minutes = now % 1 * 60
        dt = self.weather.iloc[math.floor(now)].name.to_pydatetime()
        minutes, seconds = math.floor(minutes), math.ceil(minutes % 1 * 60)
        return dt + timedelta(minutes=minutes, seconds=seconds)

    def is_workshift(self, workday_start: int = -1, workday_end: int = -1) -> bool:
        """Checks if the current simulation time is within the windfarm's working hours.

        Parameters
        ----------
        workday_start : int
            A valid hour in 24 hour time, by default -1. This should only be provided from an
            ``ServiceEquipmentData`` object. ``workday_end`` must also be provided in order to be used.
        workday_end : int
            A valid hour in 24 hour time, by default -1. This should only be provided from an
            ``ServiceEquipmentData`` object. ``workday_start`` must also be provided in order to be used.

        Returns
        -------
        bool
            True if it's valid working hours, False otherwise.
        """
        if -1 in (workday_start, workday_end):
            return self.workday_start <= self.simulation_time.hour < self.workday_end
        return workday_start <= self.simulation_time.hour < workday_end

    def hour_in_shift(
        self, hour: int, workday_start: int = -1, workday_end: int = -1
    ) -> bool:
        """Checks whether an ``hour`` is within the working hours.

        Parameters
        ----------
        hour : int
            Hour of the day.
        workday_start : int
            A valid hour in 24 hour time, by default -1. This should only be provided from an
            ``ServiceEquipmentData`` object. ``workday_end`` must also be provided in order to be used.
        workday_end : int
            A valid hour in 24 hour time, by default -1. This should only be provided from an
            ``ServiceEquipmentData`` object. ``workday_start`` must also be provided in order to be used.

        Returns
        -------
        bool
            True if ``hour`` is during working hours, False otherwise.
        """
        if -1 in (workday_start, workday_end):
            return self.workday_start <= hour < self.workday_end
        return workday_start <= hour < workday_end

    def hours_to_next_shift(self, workday_start: int = -1) -> float:
        """Time until the next work shift starts, in hours.

        Parameters
        ----------
        workday_start : int
            A valid hour in 24 hour time, by default -1. This should only be provided
            from an ``ServiceEquipmentData`` object.

        Returns
        -------
        float
            Hours until the next shift starts.
        """
        current = self.simulation_time
        start = self.workday_start if workday_start == -1 else workday_start
        if current.hour < start:
            # difference between now and workday start
            return hours_until_future_hour(current, start)
        elif current.hour == start == 0:
            # Need to manually move forward one whole day to avoid an infinite loop
            return hours_until_future_hour(current, 24)
        else:
            # time to midnight + hour of workday start
            return start + hours_until_future_hour(current, 0)

    @property
    def current_time(self) -> str:
        """Timestamp for the current time as a datetime.datetime.strftime."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    def date_ix(self, date: datetime) -> int:
        """The first index of a future date. This corresponds to the number of hours
        until this dates from the very beginning of the simulation.

        Parameters
        ----------
        date : datetime.date
            A date within the environment's simulation range.

        Returns
        -------
        int
            Index of the weather profile corresponds to the first hour of ``date``.
        """
        return np.where(self.weather.index.date == date)[0][0]

    def _weather_setup(
        self,
        weather_file: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Reads the weather data from the "<inputs>/weather" directory.

        This also fills any missing data with zeros and interpolates the values of any
        missing datetime entries.

        Parameters
        ----------
        weather_file : str
            Name of the weather file to be used by the environment. Should be contained
            within ``data_dir/weather``.
        start_year : Optional[int], optional
            Custom starting year for the weather profile, by default None. If ``None`` or
            less than the first year of the weather profile, this will be ignored.
        end_year : Optional[int], optional
            Custom ending year for the weather profile, by default None. If ``None`` or
            greater than the last year of the weather profile, this will be ignored.

        Returns
        -------
        pd.DataFrame
            The wind (and  wave) timeseries.
        """
        weather_path = os.path.join(self.data_dir, "weather", weather_file)
        weather = pd.read_csv(
            weather_path, parse_dates=["datetime"], index_col="datetime"
        )
        weather = weather.fillna(0.0)
        weather = weather.resample("H").interpolate(limit_direction="both", limit=5)

        if start_year is not None:
            weather = weather[weather.index.year >= start_year]
        if end_year is not None:
            weather = weather[weather.index.year <= end_year]

        self.start_year = weather.index.year.min()
        self.end_year = weather.index.year.max()

        return weather

    def _create_endpoints(self) -> None:
        """Creates the ``start_date`` and ``end_date`` time stamps for the simulation."""
        self.start_datetime = self.weather.index[0].to_pydatetime()
        self.end_datetime = self.weather.index[-1].to_pydatetime()

    @property
    def weather_now(self) -> Tuple[float, float]:
        """The current weather.

        Returns
        -------
        Tuple[float, float]
            Wind and wave data for the current time.
        """
        return self.weather.iloc[self.now].values

    def weather_forecast(
        self, hours: Union[int, float]
    ) -> Tuple[DatetimeIndex, np.ndarray, np.ndarray]:
        """Returns the wind and wave data for the next ``hours`` hours.

        Parameters
        ----------
        hours : Union[int, float]
            Number of hours to look ahead, rounds up to the nearest hour.

        Returns
        -------
        Tuple[DatetimeIndex, np.ndarray, np.ndarray]
            The pandas DatetimeIndex, windspeed array, and waveheight array for the
            hours requested.
        """
        start = math.ceil(self.now)
        end = start + math.ceil(hours)

        weather = self.weather.iloc[start:end]
        return (weather.index, weather.windspeed.values, weather.waveheight.values)

    def log_action(
        self,
        *,
        agent: str,
        action: str,
        reason: str,
        additional: str = "",
        system_id: str = "",
        system_name: str = "",
        part_id: str = "",
        part_name: str = "",
        system_ol: Union[float, str] = "",
        part_ol: Union[float, str] = "",
        duration: float = 0,
        request_id: str = "na",
        materials_cost: Union[int, float] = 0,
        hourly_labor_cost: Union[int, float] = 0,
        salary_labor_cost: Union[int, float] = 0,
        equipment_cost: Union[int, float] = 0,
    ) -> None:
        """Formats the logging messages into the expected format for logging.

        Parameters
        ----------
        agent : str
            Agent performing the action.
        action : str
            Action that was taken.
        reason : str
            Reason an action was taken.
        additional : str
            Any additional information that needs to be logged.
        system_id : str
            Turbine ID, ``System.id``, by default "".
        system_name : str
            Turbine name, ``System.name``, by default "".
        part_id : str
            Subassembly, component, or cable ID, ``_.id``, by default "".
        part_name : str
            Subassembly, component, or cable name, ``_.name``, by default "".
        system_ol : Union[float, str]
            Turbine operating level, ``System.operating_level``. Use an empty string for n/a, by default "".
        part_ol : Union[float, str]
            Subassembly, component, or cable operating level, ``_.operating_level``. Use
            an empty string for n/a, by default "".
        request_id : str
            The ``RepairManager`` assigned request_id found in ``RepairRequest.request_id``, by default "na".
        duration : float
            Length of time the action lasted, by default 0.
        materials_cost : Union[int, float], optional
            Total cost of materials for action, in USD, by default 0.
        hourly_labor_cost : Union[int, float], optional
            Total cost of hourly labor for action, in USD, by default 0.
        salary_labor_cost : Union[int, float], optional
            Total cost of salaried labor for action, in USD, by default 0.
        equipment_cost : Union[int, float], optional
            Total cost of equipment for action, in USD, by default 0.
        """
        self._events_logger.info(
            format_events_log_message(
                self.simulation_time,
                self.now,
                system_id,
                system_name,
                part_id,
                part_name,
                system_ol,
                part_ol,
                agent,
                action,
                reason,
                additional,
                duration,
                request_id,
                materials_cost,
                hourly_labor_cost,
                salary_labor_cost,
                equipment_cost,
            )
        )

    def _create_events_log_dataframe(self) -> pd.DataFrame:
        """Imports the logging file created in ``run`` and returns it as a formatted
        ``pandas.DataFrame``.

        Returns
        -------
        pd.DataFrame
            The formatted logging data from a simulation.
        """
        log_columns = [
            "datetime",
            "name",
            "level",
            "env_datetime",
            "env_time",
            "system_id",
            "system_name",
            "part_id",
            "part_name",
            "system_operating_level",
            "part_operating_level",
            "agent",
            "action",
            "reason",
            "additional",
            "duration",
            "request_id",
            "materials_cost",
            "hourly_labor_cost",
            "salary_labor_cost",
            "equipment_cost",
            "total_labor_cost",
            "total_cost",
        ]
        log_df = pd.read_csv(
            self.events_log_fname,
            names=log_columns,
            sep=" :: ",
            index_col=False,
            engine="python",
        )
        log_df = log_df.loc[log_df.level == "INFO"]
        log_df = log_df.drop(labels=["name", "level"], axis=1)
        log_df.datetime = pd.to_datetime(log_df.datetime)
        log_df.env_datetime = pd.to_datetime(log_df.env_datetime)
        log_df.index = log_df.datetime
        log_df = log_df.drop(labels="datetime", axis=1)
        return log_df

    def _create_operations_log_dataframe(self) -> pd.DataFrame:
        """Imports the logging file created in ``run`` and returns it as a formatted
        ``pandas.DataFrame``.

        Returns
        -------
        pd.DataFrame
            The formatted logging data from a simulation.
        """
        with open(self.operations_log_fname) as f:
            first_line = f.readline().strip().split(" :: ")
            column_names = first_line[3:]

        log_df = pd.read_csv(
            self.operations_log_fname,
            names=column_names,
            skiprows=1,
            sep=" :: ",
            index_col=False,
            engine="python",
        )

        log_df = log_df.loc[log_df.level == "INFO"]
        log_df = log_df.drop(labels=["name", "level"], axis=1)
        log_df.datetime = pd.to_datetime(log_df.datetime)
        log_df.env_datetime = pd.to_datetime(log_df.env_datetime)
        log_df.index = log_df.datetime
        log_df = log_df.drop(labels="datetime", axis=1)
        return log_df

    def convert_logs_to_csv(  # type: ignore
        self, delete_original: bool = False, return_df: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Creates a CSV file for the both the events and operations logs.

        Parameters
        ----------
        delete_original : bool, optional
            If True, delete the corresponding ".log" files, by default False.
        return_df : bool, optional
            If True, returns the pd.DataFrame objects for the operations and events
            logging, by default True.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Returns nothing if ``return_df`` is False, otherwise returns the operations and
            events log ``pd.DataFrame`` objects in that order.
        """
        events = self._create_events_log_dataframe().sort_values("env_time")
        operations = self._create_operations_log_dataframe().sort_values("env_time")

        events_fname = self.events_log_fname.replace(".log", ".csv")
        events.to_csv(events_fname, index=False)

        operations_fname = self.operations_log_fname.replace(".log", ".csv")
        operations.to_csv(operations_fname, index=False)

        if delete_original:
            os.remove(self.operations_log_fname)
            os.remove(self.events_log_fname)

        if return_df:
            return operations, events

    def power_production_potential_to_csv(  # type: ignore
        self,
        windfarm: "wombat.windfarm.Windfarm",  # type: ignore
        operations: Optional[pd.DataFrame] = None,
        return_df: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Creates the power production ``DataFrame`` and optionally returns it.

        Parameters
        ----------
        windfarm : wombat.windfarm.Windfarm
            The simulation's windfarm object.
        operations : Optional[pd.DataFrame], optional
            The operations log ``DataFrame`` if readily available, by default None. If
            ``None``, then it will be created through ``convert_logs_to_csv()``.
        return_df : bool, optional
            Indicator to return the power production for further usage, by default True.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The power potential and production timeseries data.
        """
        if operations is None:
            operations = self._create_operations_log_dataframe().sort_values("env_time")

        turbines = windfarm.turbine_id
        windspeed = self.weather.windspeed
        windspeed = windspeed.loc[operations.env_datetime]

        potential = np.vstack(
            ([windfarm.node_system(t_id).power(windspeed) for t_id in turbines])
        ).T
        potential_df = pd.DataFrame(
            [],
            index=windspeed.index,
            columns=["env_time", "env_datetime", "windspeed", "windfarm"]
            + turbines.tolist(),
        )
        potential_df[turbines] = potential
        potential_df.windspeed = windspeed.values
        potential_df.windfarm = potential.sum(axis=1)
        potential_df.env_time = operations.env_time.values
        potential_df.env_datetime = operations.env_datetime.values
        potential_df.to_csv(self.power_potential_fname, index_label="env_datetime")

        production_df = potential_df.copy()
        production_df[turbines] = (
            production_df[turbines].values * operations[turbines].values
        )
        production_df.windfarm = production_df[turbines].sum(axis=1)
        production_df.to_csv(self.power_production_fname, index_label="env_datetime")
        if return_df:
            return potential_df, production_df

    def cleanup_log_files(self, log_only=False) -> None:
        """This is a convenience method to clear the output log files in case a large
        batch of simulations is being run and there are space limitations.

        Parameters
        ----------
        log_only : bool, optional
            Only deletes the xx.log files, if True, otherwise the xx.log and xx.csv
            logging files are all deleted, by default False
        """

        os.remove(self.events_log_fname)
        os.remove(self.operations_log_fname)
        if not log_only:
            os.remove(self.events_log_fname.replace(".log", ".csv"))
            os.remove(self.operations_log_fname.replace(".log", ".csv"))

            os.remove(self.power_potential_fname)
            os.remove(self.power_production_fname)
