import pandas as pd
from models.pv_system import PvSystem
from models.bess_system import BatterySystem
from models.smart_grid import Bus 
from utils.system_parameters import *
from typing import Dict

import pandapower as pp
import pandapower.networks as nw

TIME_HORIZON = 24

def initialise_network(case: str) -> None:
    if case != 'case33bw':
        print('At the moment only case33bw has been implemented')
        raise NotImplementedError(f"Case '{case}' is not implemented.")
    return nw.case33bw()


# def run_power_flow(
#     network: pd.DataFrame,
#     load_t: list,
#     pv_power: list,
# ) -> None:
#     pp.runopp(network)
#     # Corresponding pandapower elements
#     pp.create_sgen(
#         network,
#         bus=18,
#         p_mw=0.0,
#         name="Silicon PV"
#     )


def create_pv_system(
    name: str,
    pv_parameters: Dict[str, float],
    pv_capacity: float,
    bus_idx: int,
) -> PvSystem:
    if not network:
        print('Unable to create a bess system, no network is declared')
        return
    pv = PvSystem(pv_parameters, pv_capacity)
    pp.create_sgen(
        network,
        bus=bus_idx,
        p_mw=0.0,
        name=name
    )
    return pv

def create_bess_system(
    name: str,
    bus_idx: int,
    battery_capacity: float,
    max_energy_mwh: float,
    min_energy_mwh: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    max_p_mw: float,
    min_p_mw: float,
    initial_soc: float,
) -> BatterySystem:
    if not network:
        print('Unable to create a bess system, no network initialised')
        return

    bess = BatterySystem(
        battery_capacity,
        max_energy_mwh,
        min_energy_mwh,
        max_p_mw,
        min_p_mw,
        charge_efficiency,
        discharge_efficiency,
        initial_soc,
    )

    pp.create_storage(
        network,
        bus=bus_idx,
        max_e_mwh=max_energy_mwh,
        min_e_mwh=min_energy_mwh,
        soc_percent=initial_soc,
        max_p_mw=max_p_mw,
        min_p_mw=min_p_mw,
        name=name,
    )

    return bess 

    
if __name__ == "__main__":
    network = initialise_network('case33bw')
    pv_silicon = PvSystem(
            net=network,
            bus_idx=17,
            pv_parameters=SILICON_PV_PARAMETERS,
            peak_power_kW=PV_CAPACITY,
            name="Silicon_PV",
        )

    pv_emerging = PvSystem(
        net=network,
        bus_idx=15,
        pv_parameters=EMERGING_PV_PARAMETERS,
        peak_power_kW=PV_CAPACITY,
        name="Emerging_PV",
    )

    # 3. Create BESS systems (they also add themselves to the network)
    bess_system_1 = BatterySystem(
        net=network,
        bus_idx=18,
        capacity_mwh=BATTERY_CAPACITY,
        max_energy_mwh=MAX_SOC_CHARGE,
        min_energy_mwh=MIN_SOC_CHARGE,
        charge_efficiency=CHARGE_EFFICIENCY,
        discharge_efficiency=DISCHARGE_EFFICIENCY,
        max_p_mw=2.0,
        min_p_mw=-2.0, # Assuming min_p_mw is for charging
        initial_soc_percent=0.50,
        name="BESS_1",
    )

    # Now your `network` object contains all the elements,
    # and your custom objects (`pv_silicon`, `bess_system_1`) are the
    # controllers for those elements.
    print(network)

    # Example of running a simulation step:
    # a. Calculate PV power from weather data
    # pv_power_mw = pv_silicon.power_generation(temp, irradiance) / 1000 # Convert kW to MW

    # b. Update the power in the network using your new method
    # pv_silicon.update_power(pv_power_mw)

    # c. Set a dispatch for the battery
    # bess_system_1.update_power(-1.5) # Charge at 1.5 MW

    # d. Run power flow
    pp.runpp(network)

    # e. Get results
    current_soc = bess_system_1.get_current_soc()
    print(network)
    
