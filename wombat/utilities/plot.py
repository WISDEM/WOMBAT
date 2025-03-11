"""Provides expoerimental plotting routines to help with simulation diagnostics."""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from wombat import Simulation
from wombat.windfarm import Windfarm


def plot_farm_layout(
    windfarm: Windfarm,
    figure_kwargs: dict | None = None,
    plot_kwargs: dict | None = None,
    return_fig: bool = False,
) -> None | tuple[plt.figure, plt.axes]:
    """Plot the graph representation of the windfarm as represented through WOMBAT.

    Args:
        figure_kwargs : dict, optional
            Customized keyword arguments for matplotlib figure instantiation that
            will passed as ``plt.figure(**figure_kwargs). Defaults to {}.``
        plot_kwargs : dict, optional
            Customized parameters for ``networkx.draw()`` that can will passed as
            ``nx.draw(**figure_kwargs)``. Defaults to ``with_labels=True``,
            ``horizontalalignment=right``, ``verticalalignment=bottom``,
            ``font_weight=bold``, ``font_size=10``, and ``node_color=#E37225``.
        return_fig : bool, optional
            Whether or not to return the figure and axes objects for further editing
            and/or saving. Defaults to False.

    Returns
    -------
        None | tuple[plt.figure, plt.axes]: _description_
    """
    # Set the defaults for plotting
    if figure_kwargs is None:
        figure_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    figure_kwargs.setdefault("figsize", (14, 12))
    figure_kwargs.setdefault("dpi", 200)
    plot_kwargs.setdefault("with_labels", True)
    plot_kwargs.setdefault("horizontalalignment", "right")
    plot_kwargs.setdefault("verticalalignment", "bottom")
    plot_kwargs.setdefault("font_weight", "bold")
    plot_kwargs.setdefault("font_size", 10)
    plot_kwargs.setdefault("node_color", "#E37225")

    fig = plt.figure(**figure_kwargs)
    ax = fig.add_subplot(111)

    # Get the node positions and all related edges, except the self-connected ones
    positions = {
        name: np.array([node["longitude"], node["latitude"]])
        for name, node in windfarm.graph.nodes(data=True)
    }
    edges = [el for el in windfarm.graph.edges if el[0] != el[1]]

    nx.draw(windfarm.graph, pos=positions, edgelist=edges, ax=ax, **plot_kwargs)

    fig.tight_layout()
    plt.show()

    if return_fig:
        return fig, ax
    return None


def plot_farm_availability(
    sim: Simulation,
    which: str = "energy",
    individual_turbines: bool = False,
    farm_95_CI: bool = False,
    figure_kwargs: dict | None = None,
    plot_kwargs: dict | None = None,
    legend_kwargs: dict | None = None,
    tick_fontsize: int = 12,
    label_fontsize: int = 16,
    return_fig: bool = False,
) -> tuple[plt.Figure | plt.Axes] | None:
    """Plots a line chart of the monthly availability at the wind farm level.

    Parameters
    ----------
    sim : Simulation
        A ``Simulation`` object that has been run.
    which : str
        One of "time" or "energy", to indicate the basis for the availability
        calculation, by default "energy".
    individual_turbines : bool, optional
        Indicates if faint gray lines should be added in the background for the
        availability of each turbine, by default False.
    farm_95_CI : bool, optional
        Indicates if the 95% CI area fill should be added in the background.
    figure_kwargs : dict, optional
        Custom parameters for ``plt.figure()``, by default ``figsize=(15, 7)`` and
        ``dpi=300``.
    plot_kwargs : dict, optional
        Custom parameters to be passed to ``ax.plot()``, by default a label consisting
        of the simulation name and project-level availability.
    legend_kwargs : dict, optional
        Custom parameters to be passed to ``ax.legend()``, by default ``fontsize=14``.
    tick_fontsize : int, optional
        The x- and y-axis tick label fontsize, by default 12.
    label_fontsize : int, optional
        The x- and y-axis label fontsize, by default 16.
    return_fig : bool, optional
        If ``True``, return the figure and Axes object, otherwise don't, by default
        False.

    Returns
    -------
    tuple[plt.Figure, plt.Axes] | None
        See :py:attr:`return_fig` for details.
        _description_
    """
    # Get the availability data
    metrics = sim.metrics
    if which == "time":
        availability = (
            metrics.time_based_availability("project", "windfarm")
            .astype(float)
            .values[0][0]
        )
        windfarm_availability = metrics.time_based_availability(
            "month-year", "windfarm"
        ).astype(float)
        turbine_availability = metrics.time_based_availability("month-year", "turbine")
        label = f"{sim.env.simulation_name} Time-Based Availability: {availability:.2%}"
    elif which == "energy":
        availability = (
            metrics.production_based_availability("project", "windfarm")
            .astype(float)
            .values[0][0]
        )
        windfarm_availability = metrics.production_based_availability(
            "month-year", "windfarm"
        ).astype(float)
        turbine_availability = metrics.production_based_availability(
            "month-year", "turbine"
        ).astype(float)
        label = (
            f"{sim.env.simulation_name} Energy-Based Availability: {availability:.2%}"
        )
    else:
        raise ValueError("`which` must be one of 'energy' or 'time'.")

    # Set the defaults
    if figure_kwargs is None:
        figure_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if legend_kwargs is None:
        legend_kwargs = {}
    figure_kwargs.setdefault("figsize", (15, 7))
    figure_kwargs.setdefault("dpi", 300)
    plot_kwargs.setdefault("label", label)
    legend_kwargs.setdefault("fontsize", 14)

    fig = plt.figure(**figure_kwargs)
    ax = fig.add_subplot(111)

    x = range(windfarm_availability.shape[0])

    # Individual turbine availability lines
    if individual_turbines:
        for turbine in turbine_availability.columns:
            ax.plot(
                x,
                turbine_availability[turbine].values,
                color="lightgray",
                alpha=0.75,
                linewidth=0.25,
            )

    # CI error bars surrounding the lines
    if farm_95_CI:
        N = turbine_availability.shape[1]
        Z90, Z95, Z99 = (1.64, 1.96, 2.57)  # noqa: F841
        tm = turbine_availability.values.mean(axis=1)
        tsd = turbine_availability.values.std(axis=1)
        ci_lo = tm - Z95 * (tsd / np.sqrt(N))
        ci_hi = tm + Z95 * (tsd / np.sqrt(N))
        ax.fill_between(
            x, ci_lo, ci_hi, alpha=0.5, label="95% CI for Individual Turbines"
        )

    ax.plot(
        x,
        windfarm_availability.windfarm.values,
        **plot_kwargs,
    )

    years = list(range(sim.env.start_year, sim.env.end_year + 1))
    xticks_major = [x * 12 for x in range(len(years))]
    xticks_minor = list(range(0, 12 * len(years), 3))
    xlabels_major = [f"{year:>6}" for year in years]
    xlabels_minor = ["", "Apr", "", "Oct"] * len(years)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels([f"{y:.0%}" for y in ax.get_yticks()], fontsize=tick_fontsize)

    ax.set_xlim(0, windfarm_availability.shape[0])
    ax.set_xticks(xticks_major)
    for t in ax.get_xticklabels():
        t.set_y(-0.05)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticklabels(xlabels_major, ha="left", fontsize=tick_fontsize)
    ax.set_xticklabels(xlabels_minor, minor=True, rotation=90)

    ax.set_xlabel("Simulation Time", fontsize=label_fontsize)
    ax.set_ylabel("Monthly Availability", fontsize=label_fontsize)

    ax.legend(**legend_kwargs)

    ax.grid(axis="both", which="major")
    ax.grid(alpha=0.5, which="minor")

    fig.tight_layout()
    if return_fig:
        return fig, ax  # type: ignore
    return None


def plot_operational_levels(
    sim: Simulation,
    figure_kwargs: dict | None = None,
    cbar_label_fontsize: int = 14,
    return_fig: bool = False,
):
    """Plots an hourly view of the operational levels of the wind farm and individual
    turbines as a heatmap.

    Parameters
    ----------
    sim : Simulation
        A ``Simulation`` object that has been run.
    figure_kwargs : dict, optional
        Custom settings for ``plt.figure()``, by default ``figsize=(15, 10)``
        and ``dpi=300``.
    cbar_label_fontsize : int, optional
        The default fontsize used in the color bar legend for the axis label, by default
        14.
    return_fig : bool, optional
        If ``True``, return the figure and Axes object, otherwise don't, by default
        False.

    Returns
    -------
    tuple[plt.Figure, plt.Axes] | None
        See :py:attr:`return_fig` for details.
    """
    # Set the defaults
    if figure_kwargs is None:
        figure_kwargs = {}
    figure_kwargs.setdefault("figsize", (15, 10))
    figure_kwargs.setdefault("dpi", 300)

    # Get the requisite data
    op = sim.metrics.operations
    x_ix = [
        c
        for c in op.columns
        if c not in ("env_datetime", "env_time", "year", "month", "day")
    ]
    turbines = op[x_ix].values.astype(float).T
    env_time = op.env_time.values
    env_datetime = op.env_datetime.reset_index(drop=True)

    fig = plt.figure(**figure_kwargs)
    ax = fig.add_subplot(111)

    # Create the custom color map
    bounds = [0, 0.05, 0.5, 0.75, 0.9, 0.95, 1]
    YlGnBu_discrete7 = mpl.colors.ListedColormap(
        ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"]
    )
    norm = mpl.colors.BoundaryNorm(bounds, YlGnBu_discrete7.N)

    # Plot the turbine availability
    ax.imshow(
        turbines, aspect="auto", cmap=YlGnBu_discrete7, norm=norm, interpolation="none"
    )

    # Format the y-axis
    ax.set_yticks(np.arange(len(x_ix)))
    ax.set_yticklabels(x_ix)
    ax.set_ylim(-0.5, len(x_ix) - 0.5)
    ax.hlines(
        np.arange(len(x_ix)) + 0.5,
        env_time.min(),
        env_time.max(),
        color="white",
        linewidth=1,
    )

    # Create the major x-tick data
    env_dt_yr = env_datetime.dt.year
    ix = ~env_dt_yr.duplicated()
    x_major_ticks = env_dt_yr.loc[ix].index[1:]
    x_major_ticklabels = env_dt_yr.loc[ix].values[1:]

    # Create the minor x-tick data
    env_datetime_df = env_datetime.to_frame()
    env_datetime_df["month"] = env_datetime_df.env_datetime.dt.month
    env_datetime_df["month_year"] = pd.DatetimeIndex(
        env_datetime_df.env_datetime
    ).to_period("M")
    ix = ~env_datetime_df.month_year.duplicated() & (
        env_datetime_df.month.isin((4, 7, 10))
    )
    x_minor_ticks = env_datetime_df.loc[ix].index.values
    x_minor_ticklabels = [
        "" if x == "Jul" else x
        for x in env_datetime_df.loc[ix].env_datetime.dt.strftime("%b")
    ]

    # Set the x-ticks
    ax.set_xticks(x_major_ticks)
    ax.set_xticklabels(x_major_ticklabels)
    for t in ax.get_xticklabels():
        t.set_y(-0.05)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_xticklabels(x_minor_ticklabels, minor=True, rotation=90)

    # Create a color bar legend that is propportionally spaced
    cbar = ax.figure.colorbar(
        mpl.cm.ScalarMappable(cmap=YlGnBu_discrete7, norm=norm),
        ax=ax,
        ticks=bounds,
        spacing="proportional",
        format=mpl.ticker.PercentFormatter(xmax=1),
    )
    cbar.ax.set_ylabel(
        "Availability", rotation=-90, va="bottom", fontsize=cbar_label_fontsize
    )

    ax.grid(axis="x", which="major")
    ax.grid(alpha=0.7, axis="x", which="minor")

    fig.tight_layout()
    if return_fig:
        return fig, ax
    return None
