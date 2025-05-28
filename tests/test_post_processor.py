"""Test the post-processing functionality."""

import pytest
import pandas.testing as pdt
from pytest_check import check

from wombat.core import Frequency
from wombat.core.post_processor import _check_frequency


def test_check_frequency():
    """Test `_check_frequency`."""
    assert _check_frequency("project").value == "project"
    assert _check_frequency("annual").value == "annual"
    assert _check_frequency("monthly").value == "monthly"
    assert _check_frequency("month year").value == "month_year"

    frequency = _check_frequency("project")
    assert isinstance(frequency, Frequency)
    assert frequency is Frequency.PROJECT

    with pytest.raises(ValueError):
        _check_frequency("annual", which="project")

    with pytest.raises(ValueError):
        _check_frequency("monthly", which="project")

    with pytest.raises(ValueError):
        _check_frequency("month-year", which="project")


def test_results_consistency(setup_ttp):
    """Multi-facted check to ensure consistency across metrics."""
    sim = setup_ttp
    sim.run(8760 * 5)

    metrics = sim.metrics
    ev = metrics.events

    materials = metrics.component_costs(
        "project", by_category=True
    ).materials_cost.sum()
    materials_expected = ev.materials_cost.sum()
    check.equal(materials, materials_expected)

    equipment = (
        metrics.equipment_costs("project", by_equipment=True)
        .T.rename(columns={0: "equipment_cost"})
        .convert_dtypes()
    )
    equipment.index.name = "agent"
    equipment_expected = (
        ev.loc[ev.agent.isin(sim.service_equipment), ["agent", "equipment_cost"]]
        .groupby("agent")
        .sum()
        .convert_dtypes()
    )
    pdt.assert_frame_equal(
        equipment, equipment_expected, check_dtype=False, check_index_type=False
    )

    labor = metrics.labor_costs("project", by_type=True).total_labor_cost.sum()
    labor_expected = ev.total_labor_cost.sum()
    check.equal(labor, labor_expected)

    equipment_labor = (
        metrics.equipment_labor_cost_breakdowns(
            "project", by_category=True, by_equipment=True
        )
        .droplevel("reason")
        .reset_index()
        .groupby("equipment_name")
        .sum()
    )
    pdt.assert_frame_equal(
        equipment_labor[["equipment_cost"]], equipment_expected, check_dtype=False
    )
    check.equal(equipment_labor.total_labor_cost.sum(), labor_expected)

    opex = metrics.opex("project", by_category=True).convert_dtypes()
    check.equal(opex.total_labor_cost, labor_expected)
    check.equal(opex.total_equipment_cost, equipment_expected.equipment_cost.sum())
