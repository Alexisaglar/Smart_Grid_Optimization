from os import wait
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
DELTA_T = 1

def calculate_heuristic_schedule(
    pv_forecasts: Dict[str, float],
    load_forecasts: np.array,
    bess_objects: list,
) -> pd.DataFrame:
    """
    Simulates a simple self-consumption heuristic for BESS dispatch.
    """
    bess_map = {b.name: b for b in bess_objects}
    bess_names = list(bess_map.keys())
    
    # Initialize results arrays
    p_bess_net = {name: np.zeros(24) for name in bess_names}
    soc_bess = {name: np.zeros(24) for name in bess_names}
    
    total_pv_forecast = sum(pv_forecasts.values())
    net_load = load_forecasts - total_pv_forecast

    # Initialize SoC
    for name, obj in bess_map.items():
        soc_bess[name][0] = obj.initial_soc # Assuming tuple fix is made

    # Loop through time and apply the greedy logic
    for t in range(1, 24):
        for name, obj in bess_map.items():
            # First, update SoC from previous step to current step
            soc_change = (p_bess_net[name][t-1] * 1) / obj.capacity_mwh * 100
            if p_bess_net[name][t-1] < 0: # Was charging
                soc_change /= obj.charge_efficiency
            else: # Was discharging
                soc_change *= obj.discharge_efficiency
            current_soc = soc_bess[name][t-1] - soc_change
            soc_bess[name][t] = np.clip(current_soc, 0, 100)

            # Now, make a decision for the current time 't'
            if net_load[t] < 0: # Excess PV generation
                # Charge with the excess, up to max charge power
                power_to_charge = -net_load[t]
                charge_power = min(power_to_charge, obj.max_p_mw)
                p_bess_net[name][t] = -charge_power # Negative for charging
            else: # Power deficit
                # Discharge to cover the deficit, up to max discharge power
                power_to_discharge = net_load[t]
                discharge_power = min(power_to_discharge, obj.max_p_mw)
                p_bess_net[name][t] = discharge_power # Positive for discharging

    # Build the final DataFrame
    res = {'load': load_forecasts, 'total_pv': total_pv_forecast}
    for name in bess_names:
        res[f'p_{name}'] = p_bess_net[name]
        res[f'soc_{name}'] = soc_bess[name]
    
    heuristic_df = pd.DataFrame(res)
    heuristic_df['grid_import'] = (heuristic_df['load'] - heuristic_df['total_pv'] - heuristic_df[[f'p_{name}' for name in bess_names]].sum(axis=1)).clip(lower=0)
    
    return heuristic_df

def calculate_optimal_schedule(
    pv_forecasts: Dict[str, float],
    load_forecasts: np.array,
    bess: list, 
    grid_price: list,
    forecast_range: int, 
) -> pd.DataFrame:
    bess_map = {b.name: b for b in bess}

    model = pyo.ConcreteModel("BESS Optimal Dispatch")

    # Define sets:
    model.T = pyo.RangeSet(0, forecast_range - 1) # time steps
    model.BESS_IDs = pyo.Set(initialize=bess_map.keys())
    # model.PV_IDs = pyo.Set(initialize=[pv_emerging.name, pv_silicon.name])

    # Define variables:
    # model.p_bess = pyo.Var(model.BESS_IDs, model.T, domain=pyo.Reals)
    model.p_charge = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals)
    model.p_discharge = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals)
    model.soc_bess = pyo.Var(model.BESS_IDs, model.T, domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.p_grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.p_grid_export = pyo.Var(model.T, domain=pyo.NonNegativeReals)


    # Define objective function:
    def objective_function(model):
        return sum(model.p_grid_import[t] * grid_price[t] for t in model.T)
    model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

    # Define constraints
    def power_balance_constraints(model, t):
        total_pv_gen = sum(pv_forecasts[pv_id][t] for pv_id in pv_forecasts)
        total_bess_dispatch = sum(model.p_discharge[b, t] - model.p_charge[b, t] for b in model.BESS_IDs)
        net_power = load_forecasts[t] - total_pv_gen - total_bess_dispatch
        return net_power == model.p_grid_import[t] - model.p_grid_export[t]
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_constraints)

    def soc_constraint(model, b_name, t):
        bess_obj = bess_map[b_name]
        if t == 0:
            return model.soc_bess[b_name, t] == bess_obj.initial_soc

        # soc_change = (model.soc_bess[battery, t-1] * DELTA_T) / battery.max_e_mwh * 100 
        soc_prev = model.soc_bess[b_name, t-1]
        charge_power = model.p_charge[b_name, t-1]
        discharge_power = model.p_discharge[b_name, t-1]

        # SoC change is based on POWER, not previous SoC
        soc_change_from_charge = (charge_power * bess_obj.charge_efficiency * DELTA_T) / bess_obj.capacity_mwh * 100
        soc_change_from_discharge = (discharge_power / bess_obj.discharge_efficiency * DELTA_T) / bess_obj.capacity_mwh * 100

        return model.soc_bess[b_name, t] == soc_prev + soc_change_from_charge - soc_change_from_discharge
    model.soc_evolution = pyo.Constraint(model.BESS_IDs, model.T, rule=soc_constraint)

    # CORRECTED Power Limit Constraints
    def charge_limit_constraint(model, b_name, t):
        bess_obj = bess_map[b_name]
        # Note: Your class stores max_p_mw as a tuple, this fixes it
        return model.p_charge[b_name, t] <= bess_obj.max_p_mw
    model.charge_limit = pyo.Constraint(model.BESS_IDs, model.T, rule=charge_limit_constraint)
    
    def discharge_limit_constraint(model, b_name, t):
        bess_obj = bess_map[b_name]
        return model.p_discharge[b_name, t] <= bess_obj.max_p_mw
    model.discharge_limit = pyo.Constraint(model.BESS_IDs, model.T, rule=discharge_limit_constraint)


    # Solver
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee=True)

    # Results
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        res = {'load': load_forecasts, 'grid_price': grid_price}
        for pv_id, forecast in pv_forecasts.items():
            res[f'pv_{pv_id}'] = forecast
        res['grid_import'] = [pyo.value(model.p_grid_import[t]) for t in model.T]
        res['grid_export'] = [-pyo.value(model.p_grid_export[t]) for t in model.T]

        for b_name in model.BESS_IDs:
            p_charge_vals = [pyo.value(model.p_charge[b_name, t]) for t in model.T]
            p_discharge_vals = [pyo.value(model.p_discharge[b_name, t]) for t in model.T]
            res[f'p_{b_name}'] = np.array(p_discharge_vals) - np.array(p_charge_vals)
            res[f'soc_{b_name}'] = [pyo.value(model.soc_bess[b_name, t]) for t in model.T]
            
        return pd.DataFrame(res)
    else:
        print("\nSolver failed to find an optimal solution\n")
        return None


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
    pv_forecasts_data = {
        "pv_silicon": np.concatenate([np.zeros(6), (np.random.rand(12) * 0.5), np.zeros(6)]), # 24h forecast for silicon PV
        "pv_emerging": np.concatenate([np.zeros(6), (np.random.rand(12) * 0.5), np.zeros(6)])  # 24h forecast for emerging PV
    }
    total_load_forecast = network.load.p_mw.sum() * (np.sin(np.linspace(0, 2*np.pi, 24)) * 0.4 + 0.8)
    grid_price_data = np.array([10,10,10,10,15,20,50,60,50,40,30,20] * 2) # Simple price profile

    # 4. Run the Optimization
    print("Running optimization to find the 24-hour optimal schedule...")
    bess_parameters_for_opt = {
        "bess_silicon": {
            "cap_mwh": 5.0, "max_p_mw": 1.0, "initial_soc": 50,
            "charge_eff": 0.95, "discharge_eff": 0.95
        },
        "bess_emerging": {
            "cap_mwh": 4.0, "max_p_mw": 0.8, "initial_soc": 50,
            "charge_eff": 0.92, "discharge_eff": 0.92
        }
    }
    optimal_schedule_df = calculate_optimal_schedule(
        pv_forecasts=pv_forecasts_data,
        load_forecasts=total_load_forecast,
        bess=[bess_silicon, bess_emerging],
        grid_price=grid_price_data,
        forecast_range=24,
    )

    # 5. Analyze the Optimal Schedule
    if optimal_schedule_df is not None:
        print("Optimal Schedule Found:")
        print(optimal_schedule_df.head())
        
        # Now you can plot results from the dataframe
        optimal_schedule_df[['p_silicon_bess', 'p_emerging_bess', 'grid_import']].plot(kind='line')
        plt.show()

        # And verify the results using pandapower in a loop (as discussed)



    # another example
    print("--- Running Optimization ---")
    optimal_df = calculate_optimal_schedule(
        pv_forecasts=pv_forecasts_data,
        load_forecasts=total_load_forecast,
        bess=[bess_silicon, bess_emerging],
        grid_price=grid_price_data,
        forecast_range=24,
    )

    # 2. Get the HEURISTIC (un-optimized) schedule
    print("\n--- Running Heuristic Simulation ---")
    heuristic_df = calculate_heuristic_schedule(
        pv_forecasts=pv_forecasts_data,
        load_forecasts=total_load_forecast,
        bess_objects=[bess_silicon, bess_emerging],
    )

    # 3. Calculate and Compare KPIs
    if optimal_df is not None and heuristic_df is not None:
        print("\n--- KPI Comparison ---")
        
        # Economic Comparison
        cost_optimal = (optimal_df['grid_import'] * grid_price_data).sum()
        cost_heuristic = (heuristic_df['grid_import'] * grid_price_data).sum()
        print(f"Optimal Daily Cost:   £{cost_optimal:.2f}")
        print(f"Heuristic Daily Cost: £{cost_heuristic:.2f}")
        print(f"Savings:              £{(cost_heuristic - cost_optimal):.2f}")

        # Technical Comparison
        peak_optimal = optimal_df['grid_import'].max()
        peak_heuristic = heuristic_df['grid_import'].max()
        print(f"\nOptimal Peak Grid Import:   {peak_optimal:.2f} MW")
        print(f"Heuristic Peak Grid Import: {peak_heuristic:.2f} MW")

    # 4. Visualize the Comparison (Critical for your thesis!)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    optimal_df['grid_import'].plot(ax=axes[0], label='Optimal Import', style='-')
    heuristic_df['grid_import'].plot(ax=axes[0], label='Heuristic Import', style='--')
    axes[0].set_title("Grid Import Comparison")
    axes[0].set_ylabel("Power [MW]")
    axes[0].legend()
    axes[0].grid(True)

    optimal_df['soc_silicon_bess'].plot(ax=axes[1], label='Optimal SoC', style='-')
    heuristic_df['soc_silicon_bess'].plot(ax=axes[1], label='Heuristic SoC', style='--')
    axes[1].set_title("BESS State of Charge Comparison")
    axes[1].set_ylabel("SoC [%]")
    axes[1].set_xlabel("Hour")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
