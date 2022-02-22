"""This script is used to keep the baseline results up to date with the baseline inputs.

NOTES:
    - You will need xlwings to run this script, which can be installed with: `pip install xlwings`
    - Ensure your git repository is up to date with:
        1) `git fetch origin`; replace origin with other remote/upstream as needed
        2) `git pull origin`; replace origin with other remote/upstream as needed

HOW TO:
    1) Update the library data for the baseline inputs
    2) Run the script in a wombat environment: `python run_baselines.py`
"""


from pprint import pprint
from pathlib import Path

import numpy as np
import xlwings as xw

from wombat.core import Simulation


# Initialize the Base Case Excel Book
here = Path(__file__).resolve().parent
wb = xw.Book(here / "WOMBAT_baselines.xlsx")


def compute_results_table(sim):
    """Creates the results table that gets printed to Excel for each simulation."""
    metrics = sim.metrics

    avail_time = [metrics.time_based_availability(frequency="project", by="windfarm")]
    avail_time.extend(
        metrics.time_based_availability(
            frequency="annual", by="windfarm"
        ).values.flatten()
    )
    avail_time = np.array(avail_time) * 100.0

    avail_prod = [
        metrics.production_based_availability(frequency="project", by="windfarm")
    ]
    avail_prod.extend(
        metrics.production_based_availability(
            frequency="annual", by="windfarm"
        ).values.flatten()
    )
    avail_prod = np.array(avail_prod) * 100.0

    power_production = [
        metrics.power_production(frequency="project", by_turbine=False).values[0][0]
    ]
    power_production.extend(
        metrics.power_production(frequency="annual", by_turbine=False).values.flatten()
    )
    power_production = np.array(power_production) / 1000.0

    cf_net = [metrics.capacity_factor(which="net", frequency="project", by="windfarm")]
    cf_net.extend(
        metrics.capacity_factor(
            which="net", frequency="annual", by="windfarm"
        ).values.flatten()
    )
    cf_net = np.array(cf_net) * 100.0

    cf_gross = [
        metrics.capacity_factor(which="gross", frequency="project", by="windfarm")
    ]
    cf_gross.extend(
        metrics.capacity_factor(
            which="gross", frequency="annual", by="windfarm"
        ).values.flatten()
    )
    cf_gross = np.array(cf_gross) * 100.0

    completion_rate = [metrics.task_completion_rate(which="both", frequency="project")]
    completion_rate.extend(
        metrics.task_completion_rate(which="both", frequency="annual").values.flatten()
    )
    completion_rate = np.array(completion_rate) * 100.0

    component_costs = metrics.component_costs(frequency="annual", by_category=True)
    component_costs = component_costs.droplevel("component").groupby("year").sum()
    materials = [component_costs.materials_cost.sum()]
    materials.extend(component_costs.materials_cost.values.tolist())
    materials = np.array(materials) / 1e6

    equipment = [metrics.equipment_costs(frequency="project", by_equipment=False)]
    equipment.extend(
        metrics.equipment_costs(frequency="annual", by_equipment=False).values.flatten()
    )
    equipment = np.array(equipment) / 1e6

    df = metrics.equipment_costs(frequency="annual", by_equipment=True)
    equipment_breakdown = [
        [f"{col} Cost", "Milllions USD$", df[col].sum() / 1e6, *(df[col].values / 1e6)]
        for col in df.columns
    ]

    equipment_utilization = metrics.service_equipment_utilization(frequency="project")
    df = metrics.service_equipment_utilization(frequency="annual")
    equipment_utilization = [
        [
            f"{col} Utilization",
            "%",
            equipment_utilization[col].values[0] * 100.0,
            *(df[col].values.flatten() * 100.0),
        ]
        for col in equipment_utilization.columns
    ]

    results_table = [
        [
            "Metric",
            "Units",
            "Project Total",
            *[f"Year {i}" for i in range(1, len(avail_time))],
        ],
        ["Time-based Availability", "%", *avail_time],
        ["Production-based Availability", "%", *avail_prod],
        ["Power Production", "MW", *power_production],
        ["Net Capacity Factor", "%", *cf_net],
        ["Gross Capacity Factor", "%", *cf_gross],
        ["Task Completion Rate", "%", *completion_rate],
        ["Materials Cost", "Millions USD$", *materials],
        ["Total Servicing Equipment Cost", "Millions USD$", *equipment],
        *equipment_breakdown,
        *equipment_utilization,
    ]
    pprint(results_table)
    for row in results_table:
        print(row[0], len(row))
    return results_table


# LBW Base Case


def run_lbw():
    """Runs the land-based wind base scenario and outputs the essential input and output
    data points.
    """
    print("LBW starting")
    sim_lbw = Simulation(library_path="LBW", config="base.yaml")  # type: ignore
    print("LBW data loaded")

    sim_lbw.run()
    print("LBW simulation complete")

    # Write the input data to file
    turbine_id = sim_lbw.windfarm.turbine_id[0]
    substation_id = sim_lbw.windfarm.substation_id[0]
    cable_id = (substation_id, turbine_id)

    turbine = sim_lbw.windfarm.system(turbine_id)
    substation = sim_lbw.windfarm.system(substation_id)
    cable = sim_lbw.windfarm.cable(cable_id)

    subassembly_order = [
        ["Turbine", "electrical_system", turbine.electrical_system],
        ["Turbine", "electronic_control", turbine.electronic_control],
        ["Turbine", "sensors", turbine.sensors],
        ["Turbine", "hydraulic_system", turbine.hydraulic_system],
        ["Turbine", "yaw_system", turbine.yaw_system],
        ["Turbine", "rotor_blades", turbine.rotor_blades],
        ["Turbine", "mechanical_brake", turbine.mechanical_brake],
        ["Turbine", "rotor_hub", turbine.rotor_hub],
        ["Turbine", "gearbox", turbine.gearbox],
        ["Turbine", "generator", turbine.generator],
        ["Turbine", "supporting_structure", turbine.supporting_structure],
        ["Turbine", "drive_train", turbine.drive_train],
        ["Substaion", "transformer", substation.transformer],
        ["Array Cable", "cable", cable],
    ]

    data_table = [
        [
            system,
            name,
            failure.description,
            failure.scale,
            failure.shape,
            failure.time,
            failure.materials,
            ", ".join(failure.service_equipment),
            failure.operation_reduction,
            failure.level,
        ]
        for system, name, subassembly in subassembly_order
        for failure in subassembly.data.failures.values()
    ]

    lbw_sheet = wb.sheets["LBW"]
    lbw_sheet.range("A18").expand().value = data_table

    # Write the output data to file
    lbw_sheet = wb.sheets["LBW Results"]
    results_table = compute_results_table(sim_lbw)
    lbw_sheet.range("A1").expand().value = ""
    lbw_sheet.range("A1").value = results_table

    # Cleanup
    sim_lbw.env.cleanup_log_files()


def run_osw_fixed():
    """Runs the fixed-base offshore wind base scenario and outputs the essential input
    and output data points.
    """
    print("OSW-Fixed starting")
    sim_osw_fixed = Simulation(library_path="OSW_FIXED", config="base.yaml")  # type: ignore
    print("OSW-Fixed data loaded")

    sim_osw_fixed.run()
    print("OSW-Fixed simulation complete")

    # Write the input data to file
    turbine_id = sim_osw_fixed.windfarm.turbine_id[0]
    substation_id = sim_osw_fixed.windfarm.substation_id[0]
    cable_id = (substation_id, turbine_id)

    turbine = sim_osw_fixed.windfarm.system(turbine_id)
    substation = sim_osw_fixed.windfarm.system(substation_id)
    cable = sim_osw_fixed.windfarm.cable(cable_id)

    subassembly_order = [
        ["Turbine", "electrical_system", turbine.electrical_system],
        ["Turbine", "electronic_control", turbine.electronic_control],
        ["Turbine", "sensors", turbine.sensors],
        ["Turbine", "hydraulic_system", turbine.hydraulic_system],
        ["Turbine", "yaw_system", turbine.yaw_system],
        ["Turbine", "rotor_blades", turbine.rotor_blades],
        ["Turbine", "mechanical_brake", turbine.mechanical_brake],
        ["Turbine", "rotor_hub", turbine.rotor_hub],
        ["Turbine", "gearbox", turbine.gearbox],
        ["Turbine", "generator", turbine.generator],
        ["Turbine", "supporting_structure", turbine.supporting_structure],
        ["Turbine", "drive_train", turbine.drive_train],
        ["Substaion", "transformer", substation.transformer],
        ["Array Cable", "cable", cable],
    ]

    data_table = [
        [
            system,
            name,
            failure.description,
            failure.scale,
            failure.shape,
            failure.time,
            failure.materials,
            ", ".join(failure.service_equipment),
            failure.operation_reduction,
            failure.level,
        ]
        for system, name, subassembly in subassembly_order
        for failure in subassembly.data.failures.values()
    ]

    osw_sheet = wb.sheets["OSW"]
    osw_sheet.range("A19").expand().value = data_table

    # Write the output data to file
    osw_sheet = wb.sheets["OSW Results"]
    results_table = compute_results_table(sim_osw_fixed)
    osw_sheet.range("A1").expand().value = ""
    osw_sheet.range("A1").value = results_table

    # Cleanup
    sim_osw_fixed.env.cleanup_log_files()


if __name__ == "__main__":
    run_lbw()
    run_osw_fixed()
