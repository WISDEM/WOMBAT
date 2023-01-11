"""General logging methods."""

from __future__ import annotations

import logging
import datetime
from typing import Any
from pathlib import Path
from logging.handlers import MemoryHandler


def setup_logger(
    logger_name: str, log_file: Path, level: Any = logging.INFO, capacity: int = 500
) -> None:
    """Creates the logging infrastructure for a given logging category.

    TODO: Figure out how to type check ``logging.INFO``; ``Callable``?

    Parameters
    ----------
    logger_name : str
        Name to assign to the logger.
    log_file : Path
        File name and path for where the log data should be saved.
    level : Any, optional
        Logging level, by default logging.INFO.
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s|%(message)s")
    fileHandler = logging.FileHandler(log_file, mode="w")
    fileHandler.setFormatter(formatter)

    memory_handler = MemoryHandler(capacity=capacity, target=fileHandler)
    memory_handler.setFormatter(formatter)

    logger.setLevel(level)
    # logger.addHandler(fileHandler)
    logger.addHandler(memory_handler)


def format_events_log_message(
    simulation_time: datetime.datetime,
    env_time: float,
    system_id: str,
    system_name: str,
    part_id: str,
    part_name: str,
    system_ol: float | str,
    part_ol: float | str,
    agent: str,
    action: str,
    reason: str,
    additional: str,
    duration: float,
    request_id: str,
    location: str = "na",
    materials_cost: int | float = 0,
    hourly_labor_cost: int | float = 0,
    salary_labor_cost: int | float = 0,
    equipment_cost: int | float = 0,
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
    system_ol : int | float
        System operating level, ``System.operating_level``. Use an empty string for n/a.
    part_ol : int | float
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
    location : str
        The location of where the event ocurred: should be one of site, port,
        enroute, or system, by default "na".
    materials_cost : int | float, optional
        Total cost of materials for action, in USD, by default 0.
    hourly_labor_cost : int | float, optional
        Total cost of hourly labor for action, in USD, by default 0.
    salary_labor_cost : int | float, optional
        Total cost of salaried labor for action, in USD, by default 0.
    equipment_cost : int | float, optional
        Total cost of equipment for action, in USD, by default 0.

    Returns
    -------
    str
        Formatted message for consistent logging.[summary]
    """
    total_labor_cost = hourly_labor_cost + salary_labor_cost
    total_cost = total_labor_cost + equipment_cost + materials_cost
    message = (
        f"{simulation_time}|{env_time:f}|{system_id}|{system_name}|{part_id}"
        f"|{part_name}|{system_ol:f}|{part_ol:f}|{agent}|{action}"
        f"|{reason}|{additional}|{duration:f}|{request_id}|{location}"
        f"|{materials_cost:f}|{hourly_labor_cost:f}|{salary_labor_cost:f}"
        f"|{equipment_cost:f}|{total_labor_cost:f}|{total_cost:f}"
    )
    return message
