"""Provides various utility functions that don't fit within a common theme."""

from __future__ import annotations

import math
from string import digits, punctuation
from typing import TYPE_CHECKING, Callable
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


def calculate_stack_current(
    power: np.ndarray | pd.Series,
    p1: int | float,
    p2: int | float,
    p3: int | float,
    p4: int | float,
    p5: int | float,
) -> np.ndarray | pd.Series:
    """Convert power produced, in kW to current, in amperes (I) using the efficiency
    curve from the
    `H2Integrage PEM electrolysis model <https://github.com/NREL/H2Integrate/blob/main/h2integrate/simulation/technologies/hydrogen/electrolysis/PEM_H2_LT_electrolyzer_Clusters.py>`_.

    .. math::
        i_{stack} = p1 * power^{3} + p2 * power^{2} + p3 * power + p4 * power^{1/2} + p5

    Parameters
    ----------
    power : np.ndarray | pd.Series
        Power produced, in kW.
    p1 : int | float
        First coefficient.
    p2 : int | float
        Second coefficient.
    p3 : int | float
        Third coefficient
    p4 : int | float
        Fourth coefficient.
    p5 : int | float
        Intercept.

    Returns
    -------
    np.ndarray | pd.Series
        Converted current from the farm's power production.
    """
    return p1 * power**3 + p2 * power**2 + p3 * power + p4 * np.sqrt(power) + p5


def calculate_hydrogen_production(
    rated_capacity: float,
    FE: float = 0.9999999,
    n_cells: int = 135,
    turndown_ratio: float = 0.1,
    *,
    efficiency_rate: float | None = None,
    p1: int | float | None = None,
    p2: int | float | None = None,
    p3: int | float | None = None,
    p4: int | float | None = None,
    p5: int | float | None = None,
) -> Callable:
    """Create the hydrogen production curve for an electrolyzer. One of
    :py:attr:`efficiency_rate` or a complete set of polynomial values (:py:attr:`p1`,
    :py:attr:`p2`, :py:attr:`p3`, :py:attr:`p4`, and :py:attr:`p5`) for the power
    to current conversion must be provided.

    kWh are converted to current using the following efficiency formulation when the
    polynomial inputs are all provided, and the :py:attr:`efficiency_rate` is nonzero.
    .. math::
        i_{stack} = p1 * power^{3} + p2 * power^{2} + p3 * power + p4 * power^{1/2} + p5

    Parameters
    ----------
    rated_capacity : float
        Rated maximum input power of the electrolyzer, in kW.
    FE : float, optional
        Faradic efficiency, by default 0.9999999.
    n_cells : int, optional
        Number of cells per 1 MW stack, by default 135.
    turndown_ratio : float, optional
        Minimum input power as a ratio of the rated capacity, by default 0.1.
    efficiency_rate : float, optional
        Energy efficiency in kWh per kg H2. Required if ``p1`` through ``p5 are not
        provided.
    p1 : int | float | None
        First coefficient of the efficiency polynomial curve. Required if
        :py:attr:`efficiency_rate` is not provided, by default None.
    p2 : int | float | None
        Second coefficient of the efficiency polynomial curve. Required if
        :py:attr:`efficiency_rate` is not provided, by default None.
    p3 : int | float | None
        Third coefficient of the efficiency polynomial curve. Required if
        :py:attr:`efficiency_rate` is not provided, by default None.
    p4 : int | float | None
        Fourth coefficient of the efficiency polynomial curve. Required if
        :py:attr:`efficiency_rate` is not provided, by default None.
    p5 : int | float | None
        Intercept of the efficiency polynomial curve. Required if
        :py:attr:`efficiency_rate` is not provided, by default None.

    Returns
    -------
    Callable
        Function to calculate H₂ production (kg/hr) from power input (kW).

    Raises
    ------
    ValueError
        Raised when :py:attr:`turndown_ratio` is outside the range [0, 1].
    ValueError
        Raised when :py:attr:`efficiency_rate` is provided and less than or equal to 0.
    ValueError
        Raised when neither the :py:attr:`efficiency_rate` nor polynomial values (
        :py:attr:`p1` through :py:attr:`p5`) are provided.
    ValueError
        Raised both the :py:attr:`efficiency_rate` and polynomial values (
        :py:attr:`p1` through :py:attr:`p5`) are provided.
    """
    if turndown_ratio > 1 or turndown_ratio < 0:
        msg = f"`turndown_ratio` must be between 0 and 1, not: {turndown_ratio}"
        raise ValueError(msg)

    use_efficiency = False
    if efficiency_rate is not None:
        if efficiency_rate > 0:
            use_efficiency = True
        else:
            raise ValueError("`efficiency_rate` must be a positive number.")

    poly_names = ("p1", "p2", "p3", "p4", "p5")
    poly_vals = (p1, p2, p3, p4, p5)
    invalid_polynomials = [n for n, v in zip(poly_names, poly_vals) if v is None]
    use_polynomial = len(invalid_polynomials) == 0
    if not use_efficiency and not use_polynomial:
        msg = (
            "One of `efficiency_rate` or all polynomial values (`p1`, `p2`, `p3`,"
            f" `p4`, `p5`) must be provided. Received {efficiency_rate=},"
            f" {p1=}, {p2=}, {p3=}, {p4=}, {p5=}."
        )
        raise ValueError(msg)
    if use_efficiency and use_polynomial:
        msg = (
            "Only one of `efficiency_rate` or all polynomial values (`p1`, `p2`, `p3`,"
            f" `p4`, `p5`) must be provided. Received {efficiency_rate=},"
            f" {p1=}, {p2=}, {p3=}, {p4=}, {p5=}."
        )
        raise ValueError(msg)

    F = 96485.34  # Faraday's Constant (C/mol)
    SECONDS = 3600  # seconds per hour
    molar_mass_h2 = 2.016  # molecular weight of H2 (grams/mol)
    min_power = rated_capacity * turndown_ratio

    def h2(power):
        power = np.where(power > rated_capacity, rated_capacity, power)
        h2_rate_kg_hr = np.zeros_like(power)

        if use_efficiency:
            h2_rate_kg_hr = power / efficiency_rate
        else:
            if TYPE_CHECKING:
                assert isinstance(p1, int | float)
                assert isinstance(p2, int | float)
                assert isinstance(p3, int | float)
                assert isinstance(p4, int | float)
                assert isinstance(p5, int | float)
            i = calculate_stack_current(power, p1, p2, p3, p4, p5)
            h2_rate_g_s = i * n_cells * FE * molar_mass_h2 / (2 * F)
            h2_rate_kg_hr = h2_rate_g_s * SECONDS / 1e3

        h2_rate_kg_hr[h2_rate_kg_hr < 0] = 0
        h2_rate_kg_hr[power < min_power] = 0
        return h2_rate_kg_hr

    return h2


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
