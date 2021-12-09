"""Provides various utility functions."""


import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import logging  # type: ignore
import datetime  # type: ignore
from typing import Any, Union, Callable  # type: ignore


try:  # pylint: disable=duplicate-code
    from functools import cache  # type: ignore
except ImportError:  # pylint: disable=duplicate-code
    from functools import lru_cache  # pylint: disable=duplicate-code

    cache = lru_cache(None)  # pylint: disable=duplicate-code


def convert_dt_to_hours(diff: datetime.timedelta) -> float:
    """Converts a ``datetime.timedelta`` object to number of hours at the seconds resolution.

    Parameters
    ----------
    diff : datetime.timedelta
        The difference between two ``datetime.datetime`` objects.

    Returns
    -------
    float
        Number of hours between to ``datetime.datetime`` objects.
    """
    days = diff.days * 24 if diff.days > 0 else 0
    seconds = diff.seconds / 60 / 60
    return days + seconds


def hours_until_future_hour(dt: datetime.datetime, hour: int) -> float:
    """Number of hours until a future hour in the same day for ``hour`` <= 24, otherwise,
    it is the number of hours until a time in the proceeding days.

    Parameters
    ----------
    dt : datetime.datetime
        Focal datetime.
    hour : int
        Hour that is later in the day, in 24 hour time.

    Returns
    -------
    float
        Number of hours between the two times.
    """
    if hour >= 24:
        days, hour = divmod(hour, 24)
        new_dt = dt + datetime.timedelta(days=days)
        new_dt = new_dt.replace(hour=0, minute=0, second=0)
    else:
        new_dt = dt.replace(hour=hour, minute=0, second=0)
    diff = new_dt - dt
    return convert_dt_to_hours(diff)


@cache
def _mean(*args) -> float:
    """Multiplies two numbers. Used for a reduce operation.

    Parameters
    ----------
    args : Union[int, float]
        The values to compute the mean over

    Returns
    -------
    float
        The average of the values provided
    """
    return np.mean(args)


def setup_logger(logger_name: str, log_file: str, level: Any = logging.INFO) -> None:
    """Creates the logging infrastructure for a given logging category.

    TODO: Figure out how to type check ``logging.INFO``; ``Callable``?

    Parameters
    ----------
    logger_name : str
        Name to assign to the logger.
    log_file : str
        File name and path for where the log data should be saved.
    level : Any, optional
        Logging level, by default logging.INFO.
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s"
    )
    fileHandler = logging.FileHandler(log_file, mode="w")
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)


def format_events_log_message(
    simulation_time: datetime.datetime,
    env_time: float,
    system_id: str,
    system_name: str,
    part_id: str,
    part_name: str,
    system_ol: Union[float, str],
    part_ol: Union[float, str],
    agent: str,
    action: str,
    reason: str,
    additional: str,
    duration: float,
    request_id: str,
    materials_cost: Union[int, float] = 0,
    hourly_labor_cost: Union[int, float] = 0,
    salary_labor_cost: Union[int, float] = 0,
    equipment_cost: Union[int, float] = 0,
) -> str:
    """Formats the logging messages into the expected format for logging.

    Parameters
    ----------
    simulation_time : datetime64
        Timestamp within the simulation time.
    env_time : float
        Environment simulation time (``Environment.now``).
    system_id : str
        Turbine ID, ``System.id``.
    system_name : str
        Turbine name, ``System.name``.
    part_id : str
        Subassembly, component, or cable ID, ``_.id``.
    part_name : str
        Subassembly, component, or cable name, ``_.name``.
    system_ol : Union[float, str]
        System operating level, ``System.operating_level``. Use an empty string for n/a.
    part_ol : Union[float, str]
        Subassembly, component, or cable operating level, ``_.operating_level``. Use an
        empty string for n/a.
    agent : str
        Agent performin the action.
    action : str
        Action that was taken.
    reason : str
        Reason an action was taken.
    additional : str
        Any additional information that needs to be logged.
    duration : float
        Length of time the action lasted.
    request_id : str
        The ``RepairRequest.request_id`` or "na".
    materials_cost : Union[int, float], optional
        Total cost of materials for action, in USD, by default 0.
    hourly_labor_cost : Union[int, float], optional
        Total cost of hourly labor for action, in USD, by default 0.
    salary_labor_cost : Union[int, float], optional
        Total cost of salaried labor for action, in USD, by default 0.
    equipment_cost : Union[int, float], optional
        Total cost of equipment for action, in USD, by default 0.

    Returns
    -------
    str
        Formatted message for consistent logging.[summary]
    """
    total_labor_cost = hourly_labor_cost + salary_labor_cost
    total_cost = total_labor_cost + equipment_cost + materials_cost
    message = " :: ".join(
        (
            f"{simulation_time}",
            f"{env_time}",
            f"{system_id}",
            f"{system_name}",
            f"{part_id}",
            f"{part_name}",
            f"{system_ol}",
            f"{part_ol}",
            f"{agent}",
            f"{action}",
            f"{reason}",
            f"{additional}",
            f"{duration}",
            f"{request_id}",
            f"{materials_cost}",
            f"{hourly_labor_cost}",
            f"{salary_labor_cost}",
            f"{equipment_cost}",
            f"{total_labor_cost}",
            f"{total_cost}",
        )
    )
    return message


def IEC_power_curve(
    windspeed_column: pd.Series,
    power_column: pd.Series,
    bin_width: float = 0.5,
    windspeed_start: float = 0,
    windspeed_end: float = 30.0,
) -> Callable:
    """
    Direct copy, plus bug fix from OpenOA: https://github.com/NREL/OpenOA/blob/main/operational_analysis/toolkits/power_curve/functions.py#L16-L57
    Use IEC 61400-12-1-2 method for creating wind-speed binned power curve.
    Args:
        windspeed_column (:obj:``pandas.Series``): feature column
        power_column (:obj:``pandas.Series``): response column
        bin_width(:obj:``float``): width of windspeed bin, default is 0.5 m/s according to standard
        windspeed_start(:obj:``float``): left edge of first windspeed bin
        windspeed_end(:obj:``float``): right edge of last windspeed bin
    Returns:
        :obj:``function``: Python function of type (Array[float] -> Array[float]) implementing the power curve.
    """

    # Set up evenly spaced bins of fixed width, with any value over the maximum getting np.inf
    n_bins = int(np.ceil((windspeed_end - windspeed_start) / bin_width)) + 1
    bins = np.append(np.linspace(windspeed_start, windspeed_end, n_bins), [np.inf])

    # Initialize an array which will hold the mean values of each bin
    P_bin = np.ones(len(bins) - 1) * np.nan

    # Compute the mean of each bin and set corresponding P_bin
    for ibin in range(0, len(bins) - 1):
        indices = (windspeed_column >= bins[ibin]) & (windspeed_column < bins[ibin + 1])
        P_bin[ibin] = power_column.loc[indices].mean()

    # Linearly interpolate any missing bins
    P_bin = pd.Series(data=P_bin).interpolate(method="linear").bfill().values

    # Create a closure over the computed bins which computes the power curve value for arbitrary array-like input
    def pc_iec(x):
        P = np.zeros(np.shape(x))
        for i in range(0, len(bins) - 1):
            idx = np.where((x >= bins[i]) & (x < bins[i + 1]))
            P[idx] = P_bin[i]
        cutoff_idx = (x < windspeed_start) | (x > windspeed_end)
        P[cutoff_idx] = 0.0
        return P

    return pc_iec
