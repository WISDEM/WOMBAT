"""This script is used to keep the baseline results up to date with the baseline inputs.

NOTES:
    - You will need xlwings to run this script, which can be installed with: `pip install xlwings`

HOW TO:
    1) Update the library data for the baseline inputs
    2) Run the script in a wombat environment: `python run_baselines.py`
"""


from pathlib import Path

import xlwings as xw

from wombat.core import Simulation


# Initialize the Base Case Excel Book
here = Path(__file__).resolve().parent
wb = xw.Book(here / "WOMBAT_baselines.xlsx")


# LBW Base Case


def run_lbw():
    """Runs the land-based wind base scenario and outputs the essential input and output
    data points.
    """
    print("LBW starting")
    sim_lbw = Simulation(library_path="LBW", config="base.yaml")
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

    # Cleanup
    sim_lbw.env.cleanup_log_files()


def run_osw_fixed():
    """Runs the fixed-base offshore wind base scenario and outputs the essential input
    and output data points.
    """
    print("OSW-Fixed starting")
    sim_osw_fixed = Simulation(library_path="OSW_FIXED", config="base.yaml")
    print("OSW-Fixed data loaded")

    # sim_osw_fixed.run()
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

    # Write the output data

    # Cleanup
    sim_osw_fixed.env.cleanup_log_files()


if __name__ == "__main__":
    run_lbw()
    run_osw_fixed()
