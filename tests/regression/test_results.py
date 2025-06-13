"""Test the post-processing functionality."""

import pandas.testing as pdt
from pytest_check import check


def test_results_consistency(setup_ttp):
    """Multi-facted check to ensure consistency across metrics."""
    sim = setup_ttp

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


def test_event_summary_consistency(setup_ttp):
    """Tests the consistency of request counting."""
    sim = setup_ttp
    metrics = sim.metrics

    timing_all = metrics.process_times(include_incompletes=True)
    timing_sub = metrics.process_times(include_incompletes=False).rename(
        columns={"N": "N_completed"}
    )
    requests = metrics.request_summary()

    combined = (
        timing_all[["N"]]
        .join(timing_sub[["N_completed"]], how="outer")
        .join(requests, how="outer")
        .fillna(0)
        .astype(int)
    )

    with check:
        pdt.assert_series_equal(
            combined.total_requests,
            combined.canceled_requests
            + combined.incomplete_requests
            + combined.completed_requests,
            check_names=False,
        )

    with check:
        pdt.assert_series_equal(
            combined.N,
            (combined.total_requests - combined.canceled_requests),
            check_names=False,
        )

    with check:
        pdt.assert_series_equal(
            combined.N,
            (combined.completed_requests + combined.incomplete_requests),
            check_names=False,
        )

    with check:
        pdt.assert_series_equal(
            combined.N_completed, combined.completed_requests, check_names=False
        )
