"""Provides the O&M Enviroment class; a subclass of simpy.Environment."""
from __future__ import annotations

import csv
import math
import logging
import datetime as dt
from typing import TYPE_CHECKING, Tuple, Union, Optional
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import simpy
import pandas as pd
import pyarrow as pa
import pyarrow.csv  # pylint: disable=W0611
from simpy.events import Event
from pandas.core.indexes.datetimes import DatetimeIndex

import wombat  # pylint: disable=W0611
from wombat.utilities import hours_until_future_hour
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
    """The primary mechanism for powering an O&M simulation. This object has insight into
    all other simulation objects, and controls the timing, date/time stamps, and weather
    conditions.

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
        Name of the simulation; will be used for naming the log file, by default ``None``.
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
        self.weather_dates = self.weather.index.to_pydatetime()
        self.max_run_time = self.weather.shape[0]
        self.shift_length = self.workday_end - self.workday_start

        # Set the environmental consideration parameters
        self.non_operational_start = parse_date(non_operational_start)
        self.non_operational_end = parse_date(non_operational_end)
        self.reduced_speed_start = parse_date(reduced_speed_start)
        self.reduced_speed_end = parse_date(reduced_speed_end)
        self.reduced_speed = reduced_speed

        self.simulation_name = simulation_name
        self._logging_setup()
        self.process(self._log_actions())

    def _register_windfarm(self, windfarm: Windfarm) -> None:
        """Adds the simulation windfarm to the class attributes"""
        self.windfarm = windfarm

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
        # If running a paused simulation, then reopen the file and append, but only if
        # the simulation time is lower than the upper bound
        time_check = self.now < self.max_run_time
        if self._events_csv.closed and time_check:  # type: ignore
            self._events_csv = open(self.events_log_fname, "a")
            self._events_writer = csv.DictWriter(
                self._events_csv, delimiter="|", fieldnames=EVENTS_COLUMNS
            )
        if hasattr(self, "windfarm") and self._operations_csv.closed and time_check:  # type: ignore
            self._operations_csv = open(self.operations_log_fname, "a")
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
        ix = self.weather.index.get_loc(date.strftime("%Y-%m-%d"))

        # If the index is consecutive a slice is returned, else a numpy array
        if isinstance(ix, slice):
            return ix.start
        return ix[0]

    def _weather_setup(
        self,
        weather_file: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
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
        # PyArrow datetime conversion setup
        convert_options = pa.csv.ConvertOptions(
            timestamp_parsers=[
                "%m/%d/%y %H:%M",
                "%m/%d/%y %I:%M",
                "%m/%d/%Y %H:%M",
                "%m/%d/%Y %I:%M",
            ]
        )
        weather = (
            pa.csv.read_csv(
                self.data_dir / "weather" / weather_file,
                convert_options=convert_options,
            )
            .to_pandas()
            .set_index("datetime")
        )
        weather = weather.fillna(0.0)
        weather = weather.resample("H").interpolate(limit_direction="both", limit=5)

        # Add in the hour of day column for more efficient handling within the simulation
        weather = weather.assign(hour=weather.index.hour.astype(float))

        # Create the start and end points
        self.start_datetime = weather.index[0].to_pydatetime()
        self.end_datetime = weather.index[-1].to_pydatetime()
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
            weather = weather.loc[weather.index.year >= start_year]
            self.start_datetime = weather.index[0].to_pydatetime()
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
                # Filter for the provided, validated ending year and update the attribute
                weather = weather.loc[weather.index.year <= end_year]
                self.end_datetime = weather.index[-1].to_pydatetime()
                self.end_year = self.end_datetime.year
        else:
            # Filter for the provided, validated ending year and update the attribute
            weather = weather.loc[weather.index.year <= end_year]
            self.end_datetime = weather.index[-1].to_pydatetime()
            self.end_year = self.end_datetime.year

        return weather

    @property
    def weather_now(self) -> Tuple[float, float, int]:
        """The current weather.

        Returns
        -------
        Tuple[float, float, int]
            Wind, wave, and hour data for the current time.
        """
        # Rounds down because we won't arrive at the next weather event until that hour
        now = int(self.now)
        return self.weather.iloc[now].values

    def weather_forecast(
        self, hours: Union[int, float]
    ) -> Tuple[DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the wind and wave data for the next ``hours`` hours, starting from
        the current hour's weather.

        Parameters
        ----------
        hours : Union[int, float]
            Number of hours to look ahead, rounds up to the nearest hour.

        Returns
        -------
        Tuple[DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]
            The pandas DatetimeIndex, windspeed array, and waveheight array for the
            hours requested, each with shape (``hours`` + 1).
        """
        start = math.floor(self.now)

        # If it's not on the hour, ensure we're looking ``hours`` hours into the future
        end = start + math.ceil(hours) + math.ceil(self.now % 1)

        wind, wave, hour = self.weather.values[start:end].T
        ix = self.weather.index[start:end]
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
        system_ol : float | int
            Turbine operating level, ``System.operating_level``. Use an empty string for n/a, by default 0.
        part_ol : float | int
            Subassembly, component, or cable operating level, ``_.operating_level``. Use
            an empty string for n/a, by default 0.
        request_id : str
            The ``RepairManager`` assigned request_id found in ``RepairRequest.request_id``, by default "na".
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
        row = dict(
            datetime=dt.datetime.now(),
            env_datetime=self.simulation_time,
            env_time=self.now,
            system_id=system_id,
            system_name=system_name,
            part_id=part_id,
            part_name=part_name,
            system_operating_level=system_ol,
            part_operating_level=part_ol,
            agent=agent,
            action=action,
            reason=reason,
            additional=additional,
            duration=duration,
            distance_km=distance_km,
            request_id=request_id,
            location=location,
            materials_cost=materials_cost,
            hourly_labor_cost=hourly_labor_cost,
            salary_labor_cost=salary_labor_cost,
            equipment_cost=equipment_cost,
            total_labor_cost=total_labor_cost,
            total_cost=total_cost,
        )
        self._events_buffer.append(row)

    def _log_actions(self):
        """Writes the action log items every 8000 hours"""
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
        convert_options = pa.csv.ConvertOptions(
            timestamp_parsers=["%y-%m-%d %H:%M:%S.%f", "%y-%m-%d %H:%M:%S"]
        )
        parse_options = pa.csv.ParseOptions(delimiter="|")
        log_df = pa.csv.read_csv(
            self.events_log_fname,
            convert_options=convert_options,
            parse_options=parse_options,
        ).to_pandas()
        log_df.datetime = pd.to_datetime(log_df.datetime)
        log_df.env_datetime = pd.to_datetime(log_df.env_datetime)
        log_df = log_df.set_index("datetime").sort_values("datetime")

        return log_df

    def load_operations_log_dataframe(self) -> pd.DataFrame:
        """Imports the logging file created in ``run`` and returns it as a formatted
        ``pandas.DataFrame``.

        Returns
        -------
        pd.DataFrame
            The formatted logging data from a simulation.
        """
        convert_options = pa.csv.ConvertOptions(
            timestamp_parsers=["%y-%m-%d %H:%M:%S.%f", "%y-%m-%d %H:%M:%S"]
        )
        parse_options = pa.csv.ParseOptions(delimiter="|")
        log_df = pa.csv.read_csv(
            self.operations_log_fname,
            convert_options=convert_options,
            parse_options=parse_options,
        ).to_pandas()
        log_df.datetime = pd.to_datetime(log_df.datetime)
        log_df.env_datetime = pd.to_datetime(log_df.env_datetime)
        log_df = log_df.set_index("datetime").sort_values("datetime")

        return log_df

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
        windspeed = self.weather.windspeed
        windspeed = windspeed.loc[operations.env_datetime].values
        potential_df = pd.DataFrame(
            [],
            index=operations.env_datetime,
            columns=["env_time", "env_datetime", "windspeed", "windfarm"]
            + turbines.tolist(),
        )
        potential_df[turbines] = np.vstack(
            ([windfarm.system(t_id).power(windspeed) for t_id in turbines])
        ).T
        potential_df = potential_df.assign(
            windspeed=windspeed,
            windfarm=potential_df[turbines].sum(axis=1),
            env_time=operations.env_time.values,
            env_datetime=operations.env_datetime.values,
        )
        pa.csv.write_csv(
            pa.Table.from_pandas(potential_df),
            self.power_potential_fname,
            write_options=write_options,
        )

        # TODO: The actual windfarm production needs to be clipped at each subgraph to
        # the max of the substation's operating capacity and then summed.
        production_df = potential_df.copy()
        production_df[turbines] *= operations[turbines].values
        production_df.windfarm = production_df[turbines].sum(axis=1)
        pa.csv.write_csv(
            pa.Table.from_pandas(production_df),
            self.power_production_fname,
            write_options=write_options,
        )
        if return_df:
            return potential_df, production_df

    def cleanup_log_files(self) -> None:
        """This is a convenience method to clear the output log files in case a large
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
