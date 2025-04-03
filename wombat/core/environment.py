"""Provides the O&M Enviroment class; a subclass of simpy.Environment."""

from __future__ import annotations

import io
import csv
import math
import logging
import datetime as dt
from typing import TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import simpy
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv  # pylint: disable=W0611
from simpy.events import Event

import wombat  # pylint: disable=W0611
from wombat.utilities import (
    hours_until_future_hour,
    calculate_windfarm_operational_level,
)
from wombat.core.data_classes import parse_date


if TYPE_CHECKING:
    from wombat.windfarm import Windfarm


EVENTS_COLUMNS = [
    "datetime",
    "env_datetime",
    "env_time",
    "agent",
    "action",
    "reason",
    "additional",
    "system_id",
    "system_name",
    "part_id",
    "part_name",
    "system_operating_level",
    "part_operating_level",
    "duration",
    "distance_km",
    "request_id",
    "location",
    "materials_cost",
    "hourly_labor_cost",
    "salary_labor_cost",
    "total_labor_cost",
    "equipment_cost",
    "total_cost",
]


class WombatEnvironment(simpy.Environment):
    """The primary mechanism for powering an O&M simulation. This object has insight
    into all other simulation objects, and controls the timing, date/time stamps, and
    weather conditions.

    Parameters
    ----------
    data_dir : pathlib.Path | str
        Directory where the inputs are stored and where to save outputs.
    weather_file : str
        Name of the weather file. Should be contained within ``data_dir``/weather/, with
        columns "datetime", "windspeed", and, optionally, "waveheight". The datetime
        column should adhere to the following format: "MM/DD/YY HH:MM", in 24-hour time.
    workday_start : int
        Starting time for the repair crew, in 24 hour local time. This can be overridden
        by an ``ServiceEquipmentData`` object that operates outside of the "typical"
        working hours.
    workday_end : int
        Ending time for the repair crew, in 24 hour local time. This can be overridden
        by an ``ServiceEquipmentData`` object that operates outside of the "typical"
        working hours.
    simulation_name : str | None, optional
        Name of the simulation; will be used for naming the log file, by default None.
        If ``None``, then the current time will be used. Will always save to
        ``data_dir``/outputs/logs/``simulation_name``.log.

        .. note: spaces (" ") will be replaced with underscores ("_"), for example:
            "my example analysis" becomes "my_example_analysis".

    start_year : int | None, optional
        Custom starting year for the weather profile, by default None. If ``None`` or
        less than the first year of the weather profile, this will be ignored.
    end_year : int | None, optional
        Custom ending year for the weather profile, by default None. If ``None`` or
        greater than the last year of the weather profile, this will be ignored.
    port_distance : int | float
        The simulation-wide daily travel distance for servicing equipment. This
        should be used as a base setting when multiple or all servicing equipment
        will be operating out of the same base location, but can be individually
        modified.
    non_operational_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the environment level,
        an undefined or later starting date will be overridden for all servicing
        equipment and any modeled port, by default None.
    non_operational_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of prohibited operations. When defined at the environment level,
        an undefined or earlier ending date will be overridden for all servicing
        equipment and any modeled port, by default None.
    reduced_speed_start : str | datetime.datetime | None
        The starting month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the environment level,
        an undefined or later starting date will be overridden for all servicing
        equipment and any modeled port, by default None.
    reduced_speed_end : str | datetime.datetime | None
        The ending month and day, e.g., MM/DD, M/D, MM-DD, etc. for an annualized
        period of reduced speed operations. When defined at the environment level,
        an undefined or earlier ending date will be overridden for all servicing
        equipment and any modeled port, by default None.
    reduced_speed : float
        The maximum operating speed during the annualized reduced speed operations.
        When defined at the environment level, an undefined or faster value will be
        overridden for all servicing equipment and any modeled port, by default 0.0.
    random_seed : int | None
        The random seed to be passed to a universal NumPy ``default_rng`` object to
        generate Weibull random generators, by default None.
    random_generator: np.random._generator.Generator | None
        An optional numpy random generator that can be provided to seed a simulation
        with the same generator each time, in place of the random seed. If a
        :py:attr:`random_seed` is also provided, this will override the random seed,
        by default None.

    Raises
    ------
    FileNotFoundError
        Raised if ``data_dir`` cannot be found.
    """

    def __init__(
        self,
        data_dir: Path | str,
        weather_file: str,
        workday_start: int,
        workday_end: int,
        simulation_name: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        port_distance: int | float | None = None,
        non_operational_start: str | dt.datetime | None = None,
        non_operational_end: str | dt.datetime | None = None,
        reduced_speed_start: str | dt.datetime | None = None,
        reduced_speed_end: str | dt.datetime | None = None,
        reduced_speed: float = 0.0,
        random_seed: int | None = None,
        random_generator: np.random._generator.Generator | None = None,
    ) -> None:
        """Initialization."""
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"{self.data_dir} does not exist")

        self.workday_start = int(workday_start)
        self.workday_end = int(workday_end)
        if not 0 <= self.workday_start <= 24:
            raise ValueError("workday_start must be a valid 24hr time before midnight.")
        if not 0 <= self.workday_end <= 24:
            raise ValueError("workday_end must be a valid 24hr time.")
        if self.workday_end <= self.workday_start:
            raise ValueError(
                "Work shifts must end after they start ({self.workday_start}hrs)."
            )

        self.port_distance = port_distance
        self.weather = self._weather_setup(weather_file, start_year, end_year)
        self.weather_dates = pd.DatetimeIndex(
            self.weather.get_column("datetime").to_pandas()
        ).to_pydatetime()
        self.max_run_time = self.weather.shape[0]
        self.shift_length = self.workday_end - self.workday_start

        # Set the environmental consideration parameters
        self.non_operational_start = parse_date(non_operational_start)
        self.non_operational_end = parse_date(non_operational_end)
        self.reduced_speed_start = parse_date(reduced_speed_start)
        self.reduced_speed_end = parse_date(reduced_speed_end)
        self.reduced_speed = reduced_speed

        if random_generator is not None:
            self.random_generator = random_generator
            self.random_seed = None
        elif random_seed is not None:
            self.random_seed = random_seed
            self.random_generator = np.random.default_rng(seed=random_seed)
        else:
            self.random_seed = None
            self.random_generator = np.random.default_rng()

        self.simulation_name = simulation_name
        self._logging_setup()
        self.process(self._log_actions())

    def _register_windfarm(self, windfarm: Windfarm) -> None:
        """Adds the simulation windfarm to the class attributes."""
        self.windfarm = windfarm

    def run(self, until: int | float | Event | None = None):
        """Extends the ``simpy.Environment.run`` method to change the default behavior
        if no argument is passed to ``until``, which will now run a simulation until the
        end of the weather profile is reached.

        Parameters
        ----------
        until : Optional[Union[int, float, Event]], optional
            When to stop the simulation, by default None. See documentation on
            ``simpy.Environment.run`` for more details.
        """
        # If running a paused simulation, then reopen the file and append, but only if
        # the simulation time is lower than the upper bound
        time_check = self.now < self.max_run_time
        if self._events_csv.closed and time_check:  # type: ignore
            self._events_csv = open(self.events_log_fname, "a")
            self._events_writer = csv.DictWriter(
                self._events_csv, delimiter="|", fieldnames=EVENTS_COLUMNS
            )
        if hasattr(self, "windfarm") and self._operations_csv.closed and time_check:
            self._operations_csv: io.TextIOWrapper = open(
                self.operations_log_fname, "a"
            )
            self.windfarm._setup_logger(initial=False)

        if until is None:
            until = self.max_run_time
        elif until > self.max_run_time:
            until = self.max_run_time
        try:
            super().run(until=until)
        except BaseException as e:
            # Flush the logs to so the simulation up to the point of failure is logged
            self._events_writer.writerows(self._events_buffer)
            self._events_buffer.clear()
            self._events_csv.close()
            self._operations_writer.writerows(self._operations_buffer)
            self._operations_buffer.clear()
            self._operations_csv.close()
            print(
                f"Simulation failed at hour {self.now:,.6f},"
                f" simulation time: {self.simulation_time}"
            )
            raise e

        # Ensure all logged events make it to their target file
        self._events_writer.writerows(self._events_buffer)
        self._events_buffer.clear()
        self._events_csv.close()
        self._operations_writer.writerows(self._operations_buffer)
        self._operations_buffer.clear()
        self._operations_csv.close()

    def _logging_setup(self) -> None:
        """Completes the setup for logging data."""
        if self.simulation_name is None:
            self.simulation_name = simulation = "wombat"
        else:
            simulation = self.simulation_name.replace(" ", "_")
        dt_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        events_log_fname = f"{dt_stamp}_{simulation}_events.csv"
        operations_log_fname = f"{dt_stamp}_{simulation}_operations.csv"
        power_potential_fname = f"{dt_stamp}_{simulation}_power_potential.csv"
        power_production_fname = f"{dt_stamp}_{simulation}_power_production.csv"
        metrics_input_fname = f"{dt_stamp}_{simulation}_metrics_inputs.yaml"

        log_path = self.data_dir / "results"
        if not log_path.exists():
            log_path.mkdir()
        self.events_log_fname = log_path / events_log_fname
        self.operations_log_fname = log_path / operations_log_fname
        self.power_potential_fname = log_path / power_potential_fname
        self.power_production_fname = log_path / power_production_fname
        self.metrics_input_fname = log_path / metrics_input_fname

        _dir = self.data_dir / "results"
        if not _dir.is_dir():
            _dir.mkdir()

        self._events_csv = open(self.events_log_fname, "w")
        self._operations_csv = open(self.operations_log_fname, "w")
        self._events_writer = csv.DictWriter(
            self._events_csv, delimiter="|", fieldnames=EVENTS_COLUMNS
        )
        self._events_writer.writeheader()

        self._events_buffer: list[dict] = []
        self._operations_buffer: list[dict] = []

    def get_random_seconds(self, low: int = 0, high: int = 10) -> float:
        """Generate a random number of seconds to wait, between :py:attr:`low` and
        :py:attr:`high`.

        Parameters
        ----------
        low : int, optional
            Minimum number of seconds to wait, by default 0.
        high : int, optional
            Maximum number of seconds to wait, by default 10.

        Returns
        -------
        float
            Number of seconds to wait.
        """
        seconds_to_wait, *_ = (
            self.random_generator.integers(low=low, high=high, size=1) / 3600.0
        )
        return seconds_to_wait

    @property
    def simulation_time(self) -> datetime:
        """Current time within the simulation ("datetime" column within weather)."""
        now = self.now
        minutes = now % 1 * 60
        if now == self.max_run_time:
            _dt = self.weather_dates[math.floor(now - 1)]
            _dt + timedelta(hours=1)
        else:
            _dt = self.weather_dates[math.floor(now)]

        minutes, seconds = math.floor(minutes), math.ceil(minutes % 1 * 60)
        return _dt + timedelta(minutes=minutes, seconds=seconds)

    def is_workshift(self, workday_start: int = -1, workday_end: int = -1) -> bool:
        """Check if the current simulation time is within the windfarm's working hours.

        Parameters
        ----------
        workday_start : int
            A valid hour in 24 hour time, by default -1. This should only be provided
            from an ``ServiceEquipmentData`` object. ``workday_end`` must also be
            provided in order to be used.
        workday_end : int
            A valid hour in 24 hour time, by default -1. This should only be provided
            from an ``ServiceEquipmentData`` object. ``workday_start`` must also be
            provided in order to be used.

        Returns
        -------
        bool
            True if it's valid working hours, False otherwise.
        """
        if -1 in (workday_start, workday_end):
            # Return True if the shift is around the clock
            if self.workday_start == 0 and self.workday_end == 24:
                return True
            return self.workday_start <= self.simulation_time.hour < self.workday_end

        # Return true if the shift is around the clock
        if workday_start == 0 and workday_end == 24:
            return True

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
            A valid hour in 24 hour time, by default -1. This should only be provided
            from an ``ServiceEquipmentData`` object. ``workday_end`` must also be
            provided in order to be used.
        workday_end : int
            A valid hour in 24 hour time, by default -1. This should only be provided
            from an ``ServiceEquipmentData`` object. ``workday_start`` must also be
            provided in order to be used.

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

    def date_ix(self, date: dt.datetime | dt.date) -> int:
        """The first index of a future date. This corresponds to the number of hours
        until this dates from the very beginning of the simulation.

        Parameters
        ----------
        date : datetime.datetime | datetime.date
            A date within the environment's simulation range.

        Returns
        -------
        int
            Index of the weather profile corresponds to the first hour of ``date``.
        """
        if isinstance(date, dt.datetime):
            date = date.date()
        ix, *_ = self.weather.filter(pl.col("datetime") == date)
        return ix.item()

    def _weather_setup(
        self,
        weather_file: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pl.DataFrame:
        """Reads the weather data from the "<inputs>/weather" directory, and creates the
        ``start_date`` and ``end_date`` time stamps for the simulation.

        This also fills any missing data with zeros and interpolates the values of any
        missing datetime entries.

        Parameters
        ----------
        weather_file : str
            Name of the weather file to be used by the environment. Should be contained
            within ``data_dir/weather``.
        start_year : Optional[int], optional
            Custom starting year for the weather profile, by default None. If ``None``
            or less than the first year of the weather profile, this will be ignored.
        end_year : Optional[int], optional
            Custom ending year for the weather profile, by default None. If ``None`` or
            greater than the last year of the weather profile, this will be ignored.

        Returns
        -------
        pd.DataFrame
            The wind (and  wave) timeseries.
        """
        REQUIRED = ["windspeed", "waveheight"]

        # PyArrow datetime conversion setup
        convert_options = pa.csv.ConvertOptions(
            timestamp_parsers=[
                "%m/%d/%y %H:%M",
                "%m/%d/%y %I:%M",
                "%m/%d/%y %H:%M:%S",
                "%m/%d/%y %I:%M:%S",
                "%m/%d/%Y %H:%M",
                "%m/%d/%Y %I:%M",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %I:%M:%S",
                "%m-%d-%y %H:%M",
                "%m-%d-%y %I:%M",
                "%m-%d-%y %H:%M:%S",
                "%m-%d-%y %I:%M:%S",
                "%m-%d-%Y %H:%M",
                "%m-%d-%Y %I:%M",
                "%m-%d-%Y %H:%M:%S",
                "%m-%d-%Y %I:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %I:%M",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %I:%M:%S",
            ]
        )
        weather = (
            pl.from_pandas(
                pa.csv.read_csv(
                    self.data_dir / "weather" / weather_file,
                    convert_options=convert_options,
                )
                .to_pandas()
                .fillna(0.0)
                .set_index("datetime")
                .sort_index()
                .resample("h")
                .interpolate(limit_direction="both")  # , limit=5)
                .reset_index(drop=False)
            )
            .with_row_index()
            .with_columns(
                [
                    pl.col("datetime").cast(pl.Datetime).dt.cast_time_unit("ns"),
                    (pl.col("datetime").dt.hour()).alias("hour"),
                ]
            )
        )

        missing = set(REQUIRED).difference(weather.columns)
        if missing:
            raise KeyError(
                "The weather data are missing the following required columns:"
                f" {missing}"
            )

        # Create the start and end points
        self.start_datetime = weather.get_column("datetime").dt.min()
        self.end_datetime = weather.get_column("datetime").dt.max()
        self.start_year = self.start_datetime.year
        self.end_year = self.end_datetime.year

        if start_year is None and end_year is None:
            return weather

        if start_year is None:
            pass
        elif start_year > self.end_year:
            raise ValueError(
                f"'start_year' ({start_year}) occurs after the last available year"
                f" in the weather data (range: {self.end_year})"
            )
        else:
            # Filter for the provided, validated starting year and update the attribute
            weather = (
                weather.filter(pl.col("datetime").dt.year() >= start_year)
                .drop("index")
                .with_row_index()
            )
            self.start_datetime = weather.get_column("datetime").dt.min()
            start_year = self.start_year = self.start_datetime.year

        if end_year is None:
            pass
        elif start_year is None and end_year < self.start_year:
            raise ValueError(
                f"The provided 'end_year' ({end_year}) is before the start_year"
                f" ({self.start_year})"
            )
        elif start_year is not None:
            if end_year < start_year:
                raise ValueError(
                    f"The provided 'end_year' ({end_year}) is before the start_year"
                    f" ({start_year})"
                )
            else:
                # Filter for the provided, validated ending year and update
                weather = weather.filter(pl.col("datetime").dt.year() <= end_year)
                self.end_datetime = weather.get_column("datetime").dt.max()
                self.end_year = self.end_datetime.year
        else:
            # Filter for the provided, validated ending year and update the attribute
            weather = weather.filter(pl.col("datetime").dt.year() <= end_year)
            self.end_datetime = weather.get_column("datetime").dt.max()
            self.end_year = self.end_datetime.year

        column_order = weather.columns
        column_order.insert(0, column_order.pop(column_order.index("hour")))
        column_order.insert(0, column_order.pop(column_order.index("waveheight")))
        column_order.insert(0, column_order.pop(column_order.index("windspeed")))
        column_order.insert(0, column_order.pop(column_order.index("datetime")))
        column_order.insert(0, column_order.pop(column_order.index("index")))

        # Ensure the columns are ordered correctly and re-compute pandas-compatible ix
        return weather.select(column_order).drop("index").with_row_index()

    @property
    def weather_now(self) -> pl.DataFrame:
        """The current weather.

        Returns
        -------
        pl.DataFrame
            A length 1 slice from the weather profile at the current ``int()`` rounded
            hour, in simulation time.
        """
        # Rounds down because we won't arrive at the next weather event until that hour
        now = int(self.now)
        return self.weather.slice(now, 1)

    def weather_forecast(
        self, hours: int | float
    ) -> tuple[pl.Series, pl.Series, pl.Series, pl.Series]:
        """Returns the datetime, wind, wave, and hour data for the next ``hours`` hours,
        starting from the current hour's weather.

        Parameters
        ----------
        hours : Union[int, float]
            Number of hours to look ahead, rounds up to the nearest hour.

        Returns
        -------
        tuple[pl.Series, pl.Series, pl.Series, pl.Series]
            Each of the relevant columns (datetime, wind, wave, hour) from the weather
            profile.
        """
        # If it's not on the hour, ensure we're looking ``hours`` hours into the future
        start = math.floor(self.now)
        _, ix, wind, wave, hour, *_ = self.weather.slice(
            start, math.ceil(hours) + math.ceil(self.now % 1)
        )
        return ix, hour, wind, wave

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
        system_ol: float | int = 0,
        part_ol: float | int = 0,
        duration: float = 0,
        distance_km: float = 0,
        request_id: str = "na",
        location: str = "na",
        materials_cost: int | float = 0,
        hourly_labor_cost: int | float = 0,
        salary_labor_cost: int | float = 0,
        equipment_cost: int | float = 0,
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
        system_ol : float | int
            Turbine operating level, ``System.operating_level``. Use an empty string
            for n/a, by default 0.
        part_ol : float | int
            Subassembly, component, or cable operating level, ``_.operating_level``. Use
            an empty string for n/a, by default 0.
        request_id : str
            The ``RepairManager`` assigned request_id found in
            ``RepairRequest.request_id``, by default "na".
        location : str
            The location of where the event ocurred: should be one of site, port,
            enroute, or system, by default "na".
        duration : float
            Length of time the action lasted, by default 0.
        distance : float
            Distance traveled, in km, if applicable, by default 0.
        materials_cost : Union[int, float], optional
            Total cost of materials for action, in USD, by default 0.
        hourly_labor_cost : Union[int, float], optional
            Total cost of hourly labor for action, in USD, by default 0.
        salary_labor_cost : Union[int, float], optional
            Total cost of salaried labor for action, in USD, by default 0.
        equipment_cost : Union[int, float], optional
            Total cost of equipment for action, in USD, by default 0.
        """
        valid_locations = ("site", "system", "port", "enroute", "na")
        if location not in valid_locations:
            raise ValueError(
                f"Event logging `location` must be one of: {valid_locations}"
            )
        total_labor_cost = hourly_labor_cost + salary_labor_cost
        total_cost = total_labor_cost + equipment_cost + materials_cost
        now = self.simulation_time
        row = {
            "datetime": dt.datetime.now(),
            "env_datetime": now,
            "env_time": self.now,
            "system_id": system_id,
            "system_name": system_name,
            "part_id": part_id,
            "part_name": part_name,
            "system_operating_level": system_ol,
            "part_operating_level": part_ol,
            "agent": agent,
            "action": action,
            "reason": reason,
            "additional": additional,
            "duration": duration,
            "distance_km": distance_km,
            "request_id": request_id,
            "location": location,
            "materials_cost": materials_cost,
            "hourly_labor_cost": hourly_labor_cost,
            "salary_labor_cost": salary_labor_cost,
            "equipment_cost": equipment_cost,
            "total_labor_cost": total_labor_cost,
            "total_cost": total_cost,
        }
        # Don't log the initiation of a crew transfer that can forced at the end of an
        # operation but happens to be after the end of the simulation
        if now <= self.end_datetime:
            self._events_buffer.append(row)

    def _log_actions(self):
        """Writes the action log items every 8000 hours."""
        HOURS = 8000
        while True:
            yield self.timeout(HOURS)
            self._events_writer.writerows(self._events_buffer)
            self._events_buffer.clear()

    def load_events_log_dataframe(self) -> pd.DataFrame:
        """Imports the logging file created in ``run`` and returns it as a formatted
        ``pandas.DataFrame``.

        Returns
        -------
        pd.DataFrame
            The formatted logging data from a simulation.
        """
        log_df = (
            pd.read_csv(
                self.events_log_fname,
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
        return log_df

    def _calculate_adjusted_production(
        self, op: pd.DataFrame, prod: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculates the overall wind farm power production and adjusts individual
        turbine production by accounting for substation downtime. This is done by
        multiplying the all downstream turbine operational levels by the substation's
        operational level.

        Parameters
        ----------
        op : pd.DataFrame
            The operational level DataFrame with turbine, substation, and windfarm
            columns.
        prod : pd.DataFrame
            The turbine energy production DataFrame.

        Notes
        -----
        This is a crude cap on the operations, and so a smarter way of capping
        the availability should be added in the future.

        Returns
        -------
        pd.DataFrame
            Either the aggregate wind farm operational level or the total wind farm
            energy production if the :py:attr:`prod` is provided.
        """
        # Adjust individual turbine production for substation downtime
        prod = prod.copy()
        for sub, val in self.windfarm.substation_turbine_map.items():
            prod[val["turbines"]] *= op[[sub]].values
        prod.windfarm = prod[self.windfarm.turbine_id].sum(axis=1)
        return prod[["windfarm"]]

    def load_operations_log_dataframe(self) -> pd.DataFrame:
        """Imports the logging file created in ``run`` and returns it as a formatted
        ``pandas.DataFrame``.

        Returns
        -------
        pd.DataFrame
            The formatted logging data from a simulation.
        """
        log_df = (
            pd.read_csv(
                self.operations_log_fname,
                delimiter="|",
                engine="pyarrow",
            )
            .set_index("datetime")
            .sort_values("datetime")
        )

        # Adjust the turbine operational values to account for substation downtime
        for sub, val in self.windfarm.substation_turbine_map.items():
            log_df[val["turbines"]] *= log_df[[sub]].values

        # Calculate the wind farm aggregate operational value
        log_df["windfarm"] = calculate_windfarm_operational_level(
            operations=log_df,
            turbine_id=self.windfarm.turbine_id,
            turbine_weights=self.windfarm.turbine_weights,
            substation_turbine_map=self.windfarm.substation_turbine_map,
        )
        return log_df

    def power_production_potential_to_csv(  # type: ignore
        self,
        windfarm: wombat.windfarm.Windfarm,
        operations: pd.DataFrame | None = None,
        return_df: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Creates the power production ``DataFrame`` and optionally returns it.

        Parameters
        ----------
        windfarm : wombat.windfarm.Windfarm
            The simulation's windfarm object.
        operations : Optional[pd.DataFrame], optional
            The operations log ``DataFrame`` if readily available, by default None. If
            ``None``, then it will be created through
            ``load_operations_log_dataframe()``.
        return_df : bool, optional
            Indicator to return the power production for further usage, by default True.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The power potential and production timeseries data.
        """
        write_options = pa.csv.WriteOptions(delimiter="|")

        if operations is None:
            operations = self.load_operations_log_dataframe().sort_values("env_time")

        turbines = windfarm.turbine_id
        windspeed = self.weather.to_pandas().set_index("datetime").windspeed
        windspeed = windspeed.loc[operations.env_datetime].values
        potential_df = pd.DataFrame(
            [],
            index=operations.env_datetime,
            columns=["env_time", "env_datetime", "windspeed", "windfarm"]
            + turbines.tolist(),
        )
        potential_df[turbines] = np.vstack(
            [windfarm.system(t_id).power(windspeed) for t_id in turbines]
        ).T
        potential_df = potential_df.assign(
            windspeed=windspeed,
            windfarm=potential_df[turbines].sum(axis=1),
            env_time=operations.env_time.values,
            env_datetime=operations.env_datetime.values,
        )
        pa.csv.write_csv(
            pa.Table.from_pandas(potential_df, preserve_index=False),
            self.power_potential_fname,
            write_options=write_options,
        )

        # TODO: The actual windfarm production needs to be clipped at each subgraph to
        # the max of the substation's operating capacity and then summed.
        production_df = potential_df.copy()
        production_df[turbines] *= operations[turbines].values
        production_df.windfarm = self._calculate_adjusted_production(
            operations, production_df
        )
        pa.csv.write_csv(
            pa.Table.from_pandas(production_df, preserve_index=False),
            self.power_production_fname,
            write_options=write_options,
        )
        if return_df:
            return potential_df, production_df

    def cleanup_log_files(self) -> None:
        """Convenience method to clear the output log files in case a large
        batch of simulations is being run and there are space limitations.

        ... warning:: This shuts down the loggers, so no more logging will be able
            to be performed.
        """
        # NOTE: Everything is wrapped in a try/except clause to protect against failure
        # when inevitably a file has already been deleted on accident, or if in the
        # dataframe generation step, the original logs were deleted

        logging.shutdown()
        if not self._events_csv.closed:
            self._events_csv.close()
        if not self._operations_csv.closed:
            self._operations_csv.close()

        try:
            self.events_log_fname.unlink()
        except FileNotFoundError:
            pass

        try:
            self.operations_log_fname.unlink()
        except FileNotFoundError:
            pass

        try:
            self.power_potential_fname.unlink()
        except FileNotFoundError:
            pass

        try:
            self.power_production_fname.unlink()
        except FileNotFoundError:
            pass

        try:
            self.metrics_input_fname.unlink()
        except FileNotFoundError:
            pass
