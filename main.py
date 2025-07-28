from pandapower import optimal_powerflow
import pyomo.environ as pyo
import numpy as np
import pandas as pd
from models.pv_system import PvSystem
from models.bess_system import BatterySystem
from models.smart_grid import Bus 
from utils import pv_parameters
from utils.system_parameters import *
from typing import Dict
from pandapower.plotting.plotly import pf_res_plotly

import pandapower as pp
import pandapower.networks as nw

import matplotlib.pyplot as plt

TIME_HORIZON = 24

def calculate_optimal_schedule(
    pv_forecasts: Dict[str, float],
    load_forecasts: np.array,
    grid_price: list,
    forecast_range: int, 
) -> pd.DataFrame:
    model = pyo.ConcreteModel("BESS Optimal Dispatch")

    # Define sets:
    model.T = pyo.RangeSet(0, forecast_range - 1) # time steps
    model.BESS_IDs = pyo.Set(initialize=[bess_emerging.name, bess_silicon.name])
    # model.PV_IDs = pyo.Set(initialize=[pv_emerging.name, pv_silicon.name])

    # Define variables:
    model.p_bess = pyo.Var(model.BESS_IDs, model.T, domain=pyo.Reals)
    model.soc_bess = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.p_grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.p_grid_export = pyo.Var(model.T, domain=pyo.NonNegativeReals)


    # Define objective function:
    def objective_function(model):
        return sum(model.p_grid[t] * grid_price[t] for t in model.T)
    model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize))

    # Define constraints
    def power_balance_constraints(model, t):
        return
        # total_pv_gen = sum(pv_forecasts[pv_id][t] for pv_id in pv_forecasts)
        # total_bess_dispatch = sum(model.p_bess[bess_id][t] for bess_id in model.BESS_IDs)
        # net_power = load_forecasts[t] - total_pv_gen - total_bess_dispatch
        #    

    return 


def initialise_network(case: str) -> None:
    if case != 'case33bw':
        print('At the moment only case33bw has been implemented')
        raise NotImplementedError(f"Case '{case}' is not implemented.")
    return nw.case33bw()

def create_pv_system(
    network: pp.pandapowerNet,
    bus_idx: int,
    pv_parameters: Dict[str, float],
    name: str,
) -> PvSystem:
    if not network:
        print('Unable to create a bess system, no network is declared')
        return

    pv = PvSystem(
        network=network,
        bus_idx=bus_idx,
        pv_parameters=pv_parameters,
        name=name,
    )

    return pv

def create_bess_system(
    network: pp.pandapowerNet,
    bus_idx: int,
    initial_soc: float,
    name: str,
) -> BatterySystem:
 
    if not network:
        print('Unable to create a bess system, no network initialised')
        return

    bess = BatterySystem(
        network=network,
        bus_idx=bus_idx,
        capacity_mwh=BATTERY_CAPACITY,
        max_energy_mwh=MAX_SOC_CHARGE,
        min_energy_mwh=MIN_SOC_CHARGE,
        charge_efficiency=CHARGE_EFFICIENCY,
        discharge_efficiency=DISCHARGE_EFFICIENCY,
        max_p_mw=MAX_P_BESS,
        min_p_mw=-MIN_P_BESS,
        initial_soc_percent=initial_soc,
        name=name,
    )
    return bess 

    
if __name__ == "__main__":
    network = initialise_network('case33bw')

    pv_silicon = PvSystem(
        network=network,
        bus_idx=17,
        pv_parameters=SILICON_PV_PARAMETERS,
        name="silicon_pv",
    )
    pv_emerging = PvSystem(
        network=network,
        bus_idx=16,
        pv_parameters=EMERGING_PV_PARAMETERS,
        name="emerging_pv",
    )

    bess_emerging = create_bess_system(
        network=network,
        bus_idx=16,
        initial_soc=0.5,
        name="emerging_bess"
    )
    bess_silicon = create_bess_system(
        network=network,
        bus_idx=17,
        initial_soc=0.5,
        name="silicon_bess"
    )

    pv_silicon.update_generation(25, 700)
    pv_emerging.update_generation(25, 700)

    calculate_optimal_schedule([1,3,5],[0,1,2,3], [0,4,5], 24)
    # d. Run power flow
    # pp.runopp(network, verbose=True)
    # # print(network.sgen.pv_emerging)
    # plt.plot(network.res_bus.p_mw[1:])
    # plt.show()
    # plt.plot(network.res_bus.vm_pu[1:])
    # plt.show()
