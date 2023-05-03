"""General methods for time-based calculations."""

from __future__ import annotations

import datetime

from dateutil.parser import parse


HOURS_IN_DAY = 24
HOURS_IN_YEAR = 8760


def parse_date(value: str | None | datetime.datetime) -> datetime.datetime | None:
    """Thin wrapper for ``dateutil.parser.parse`` that converts string dates and returns
    back None or the original value if it's None or a ``datetime.datetime`` object,
    respectively.

    Parameters
    ----------
    value : str | None | datetime.datetime
        A month/date or month-date formatted string to be converted to a
        ``datetime.datetime`` object, or ``datetime.datetime`` object, or None.

    Returns
    -------
    datetime.datetime | None
        A converted ``datetime.datetime`` object or None.
    """
    if value is None:
        return value
    if isinstance(value, datetime.datetime):
        return value

    # Ensure there is a common comparison year for all datetime values
    return parse(value).replace(year=2022)


def convert_dt_to_hours(diff: datetime.timedelta) -> float:
    """Convert a ``datetime.timedelta`` object to number of hours at the seconds
    resolution.

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
    """Number of hours until a future hour in the same day for ``hour`` <= 24,
    otherwise, it is the number of hours until a time in the proceeding days.

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

        # Convert to python int
        days = int(days)
        hour = int(hour)

        new_dt = dt + datetime.timedelta(days=days)
        new_dt = new_dt.replace(hour=hour, minute=0, second=0)
    else:
        new_dt = dt.replace(hour=hour, minute=0, second=0)
    diff = new_dt - dt
    return convert_dt_to_hours(diff)


def check_working_hours(
    env_start: int, env_end: int, workday_start: int, workday_end: int
) -> tuple[int, int]:
    """Checks the working hours of a port or servicing equipment, and overrides a
    default (-1) to the environment's settings, otherwise returns back the input hours.

    Parameters
    ----------
    env_start : int
        The starting hour for the environment's shift
    env_end : int
        The ending hour for the environment's shift
    workday_start : int
        The starting hour to be checked.
    workday_end : int
        The ending hour to be checked.

    Returns
    -------
    tuple[int, int]
        The starting and ending hour to be applied back to the port or servicing
        equipment.
    """
    start_is_invalid = workday_start == -1
    end_is_invalid = workday_end == -1

    start = env_start if start_is_invalid else workday_start
    end = env_end if end_is_invalid else workday_end

    return start, end


def calculate_cost(
    duration: int | float, rate: float, n_rate: int = 1, daily_rate: bool = False
) -> float:
    """Calculates the equipment cost, or labor cost for either salaried or hourly
    employees.

    Parameters
    ----------
    duration : int | float
        Length of time, in hours.
    rate : float
        The labor or equipment rate, in $USD/hour.
    n_rate : int
        Total number of of the rate to be applied, more than one if``rate`` is broken
        down by number of individual laborers (if rate is a labor rate), by default 1.
    daily_rate : bool, optional
        Indicator for if the ``rate`` is a daily rate (True), or hourly rate (False), by
        default False.

    Returns
    -------
    float
        The total cost of the labor performed.
    """
    multiplier = duration / HOURS_IN_DAY if daily_rate else duration
    return multiplier * rate * n_rate
