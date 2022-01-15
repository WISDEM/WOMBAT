"""Test the ServiceEquipment class."""

from wombat.core import RepairManager
from wombat.windfarm import Windfarm

from tests.conftest import env_setup


def test_service_equipment_init(env_setup):
    env = env_setup
    manager = RepairManager(env)
    windfarm = Windfarm(env, "layout.csv", manager)
