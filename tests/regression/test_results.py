"""Test the post-processing functionality."""

import pandas.testing as pdt
from pytest_check import check


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
    check.almost_equal(materials, materials_expected)

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
    with check:
        pdt.assert_frame_equal(
            equipment, equipment_expected, check_dtype=False, check_index_type=False
        )

    labor = metrics.labor_costs("project", by_type=True).total_labor_cost.sum()
    labor_expected = ev.total_labor_cost.sum()
    check.almost_equal(labor, labor_expected)

    equipment_labor = (
        metrics.equipment_labor_cost_breakdowns(
            "project", by_category=True, by_equipment=True
        )
        .droplevel("reason")
        .reset_index()
        .groupby("equipment_name")
        .sum()
    )
    eql_subset = equipment_labor.loc[
        equipment_labor.equipment_cost > 0, ["equipment_cost"]
    ]
    eql_subset.index.name = "agent"
    with check:
        pdt.assert_frame_equal(eql_subset, equipment_expected, check_dtype=False)
    check.almost_equal(equipment_labor.total_labor_cost.sum(), labor_expected)

    opex = metrics.opex("project", by_category=True).convert_dtypes()
    check.almost_equal(opex.total_labor_cost.squeeze(), labor_expected)
    check.almost_equal(
        opex.equipment_cost.squeeze(), equipment_expected.equipment_cost.sum()
    )
