import pandas as pd
from models.pv_system import PvSystem
from models.bess_system import BatterySystem
from models.smart_grid import Bus 
from utils.system_parameters import *

import pandapower as pp
import pandapower.networks as nw

if __name__ == "__main__":
    network = nw.case33bw()
    pv_system_emerging = PvSystem(EMERGING_PV_PARAMETERS, PV_CAPACITY)
    pv_system_silicon = PvSystem(SILICON_PV_PARAMETERS, PV_CAPACITY)

    bess_system_silicon = BatterySystem(
        BATTERY_CAPACITY,
        MAX_SOC_CHARGE,
        MIN_SOC_CHARGE,
        CHARGE_EFFICIENCY,
        DISCHARGE_EFFICIENCY,
        2,
    )

    bess_system_emerging = BatterySystem(
        BATTERY_CAPACITY,
        MAX_SOC_CHARGE,
        MIN_SOC_CHARGE,
        CHARGE_EFFICIENCY,
        DISCHARGE_EFFICIENCY,
        4,
    )

    # Correspoding pandapower elements
    pv_idx_silicon = pp.create_sgen(
        network,
        bus=18,
        p_mw=0.0,
        name="Silicon PV"
    )

    pv_idx_emerging = pp.create_sgen(
        network,
        bus=18,
        p_mw=0.0,
        name="Emerging PV"
    )

    bess_idx_silicon = pp.create_storage(
        network,
        bus=18,
        p_mw=0.0,
        max_e_mwh=bess_system_silicon.max_soc,

    )
