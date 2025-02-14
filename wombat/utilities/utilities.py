"""Provides various utility functions that don't fit within a common theme."""

from __future__ import annotations

import math
from string import digits, punctuation
from typing import Callable
from functools import cache

import numpy as np
import pandas as pd


# Don't repeat the most common inputs that occur when there are no state changes, but
# the result is needed again for logging
@cache
def _mean(*args) -> float:
    """Multiplies two numbers. Used for a reduce operation.

    Parameters
    ----------
    args : int | float
        The values to compute the mean over

    Returns
    -------
    float
        The average of the values provided
    """
    return np.mean(args)


def create_variable_from_string(string: str) -> str:
    """Creates a valid Python variable style string from a passed string.

    Parameters
    ----------
    string : str
        The string to convert into a Python-friendly variable name.

    Returns
    -------
    str
        A Python-valid variable name.

    Examples
    --------
    >>> example_string = "*Electrical!*_ _System$*_"
    >>> print(create_variable_from_string(example_string))
    'electrical_system'

    """
    new_string = (
        string.lstrip(punctuation + digits)
        .translate(str.maketrans("", "", punctuation.replace("_", " ")))
        .lower()
    )
    return new_string


def IEC_power_curve(
    windspeed_column: np.ndarray | pd.Series,
    power_column: np.ndarray | pd.Series,
    bin_width: float = 0.5,
    windspeed_start: float = 0.0,
    windspeed_end: float = 30.0,
) -> Callable:
    """Direct copyfrom OpenOA:
    https://github.com/NREL/OpenOA/blob/main/operational_analysis/toolkits/power_curve/functions.py#L16-L57
    Use IEC 61400-12-1-2 method for creating wind-speed binned power curve.

    Parameters
    ----------
        windspeed_column : np.ndarray | pandas.Series
            The power curve's windspeed values, in m/s.
        power_column : np.ndarray | pandas.Series
            The power curve's output power values, in kW.
        bin_width : float
            Width of windspeed bin, default is 0.5 m/s according to standard, by default
            0.5.
        windspeed_start : float
            Left edge of first windspeed bin, where all proceeding values will be 0.0,
            by default 0.0.
        windspeed_end : float
            Right edge of last windspeed bin, where all following values will be 0.0, by
            default 30.0.

    Returns
    -------
        Callable
            Python function of the power curve, of type (Array[float] -> Array[float]),
            that maps input windspeed value(s) to ouptut power value(s).
    """
    if not isinstance(windspeed_column, pd.Series):
        windspeed_column = pd.Series(windspeed_column)
    if not isinstance(power_column, pd.Series):
        power_column = pd.Series(power_column)

    # Set up evenly spaced bins of fixed width, with np.inf for any value over the max
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

    # Create a closure over the computed bins which computes the power curve value for
    # arbitrary array-like input
    def pc_iec(x):
        P = np.zeros(np.shape(x))
        for i in range(0, len(bins) - 1):
            idx = np.where((x >= bins[i]) & (x < bins[i + 1]))
            P[idx] = P_bin[i]
        cutoff_idx = (x < windspeed_start) | (x > windspeed_end)
        P[cutoff_idx] = 0.0
        return P

    return pc_iec


def calculate_windfarm_operational_level(
    operations: pd.DataFrame,
    turbine_id: np.ndarray | list[str],
    turbine_weights: pd.DataFrame,
    substation_turbine_map: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Calculates the overall wind farm operational level, accounting for substation
    downtime by multiplying the sum of all downstream turbine operational levels by
    the substation's operational level.

    Parameters
    ----------
    operations : pd.DataFrame
        The turbine and substation operational level DataFrame.
    turbine_id : np.ndarray | list[str]
        The turbine ids that match :py:attr:`Windfarm.turbine_id`.
    turbine_weights : pd.DataFrame
        The turbine weights, coming from :py:attr:`Windfarm.turbine_weights`.
    substation_turbine_map : dict[str, dict[str, np.ndarray]]
        The :py:attr:`Windfarm.substation_turbine_map`.

    Notes
    -----
    This is a crude cap on the operations, and so a smarter way of capping
    the availability should be added in the future.

    Returns
    -------
    pd.DataFrame
        The aggregate wind farm operational level.
    """
    turbines = turbine_weights[turbine_id].values * operations[turbine_id]
    total = np.sum(
        [
            np.array(
                [[math.fsum(row)] for _, row in turbines[val["turbines"]].iterrows()]
            ).reshape(-1, 1)
            for sub, val in substation_turbine_map.items()
        ],
        axis=0,
    )
    return total
